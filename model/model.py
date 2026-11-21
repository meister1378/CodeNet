"""CoDE-Net (Crowd of Depth Estimation Net): single-network crowd head localization + camera geometry +
per-person metric BEV lift.

Design notes:
  * Backbone: own CSP/ELAN-style stages built from QARepVGG-style RepConv (3x3+BN | 1x1 | identity -> post-BN),
    BN + ReLU only (no LayerNorm / GELU / depthwise) for TensorRT FP16/INT8 friendliness.
    First two downsamplings use space-to-depth + conv (no information loss for tiny heads).
  * One ReLU linear-attention block at stride 32 only (global crowd / scene context, cheap).
  * Thin top-down neck P5->P4->P3->P2 with per-pixel scale-selection fusion (softmax weights, not concat).
  * Point heads: stride 8 (K=4 reference points / cell) and stride 4 (K=1 / cell); density gate picks one
    head per region (static-shape version of PET's quadtree). In-graph 3x3 max-pool suppression + top-K.
  * Camera head: global pooled S5 -> (log-FoV, pitch, roll, log camera height).
  * Per-person head size (log, per stride) -> crowd-based self-calibration of a fixed camera (horizon, height).
  * Per-person metric log-distance + Laplace scale. Lift: gravity-aligned pixel ray; distance = inverse-variance
    fusion of the ray/head-plane intersection (precise when the ray is steep) and the learned distance (only
    usable cue near / above the horizon, where the plane intersection diverges). Far people stay valid.
  * Training-only auxiliary heads (dense log-depth, up-vector field, density) are dropped in deploy mode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- basic blocks
class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=1, s=1, act=True):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout))
        if act:
            self.add_module("act", nn.ReLU(inplace=True))


class RepConv(nn.Module):
    """QARepVGG-style block: 3x3 conv-BN + 1x1 conv + identity, summed, then post-BN + ReLU.

    Only the 3x3 branch has its own BN; 1x1 and identity are BN-free (keeps the fused kernel's
    weight distribution quantization friendly). `reparameterize()` collapses it to one 3x3 conv.
    """

    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.cin, self.cout, self.s = cin, cout, s
        self.conv3 = nn.Conv2d(cin, cout, 3, s, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.conv1 = nn.Conv2d(cin, cout, 1, s, 0, bias=False)
        self.has_id = cin == cout and s == 1
        self.post_bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.fused = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        y = self.bn3(self.conv3(x)) + self.conv1(x)
        if self.has_id:
            y = y + x
        return self.act(self.post_bn(y))

    @torch.no_grad()
    def reparameterize(self):
        def fold(w, b, bn):
            std = (bn.running_var + bn.eps).sqrt()
            t = bn.weight / std
            return w * t.reshape(-1, 1, 1, 1), bn.bias + (b - bn.running_mean) * t

        w3, b3 = fold(self.conv3.weight, torch.zeros(self.cout, device=self.conv3.weight.device), self.bn3)
        w = w3 + F.pad(self.conv1.weight, [1, 1, 1, 1])
        b = b3
        if self.has_id:
            idk = torch.zeros_like(w)
            for i in range(self.cout):
                idk[i, i, 1, 1] = 1.0
            w = w + idk
        w, b = fold(w, b, self.post_bn)
        conv = nn.Conv2d(self.cin, self.cout, 3, self.s, 1, bias=True).to(w.device)
        conv.weight.copy_(w)
        conv.bias.copy_(b)
        self.fused = conv
        for n in ["conv3", "bn3", "conv1", "post_bn"]:
            delattr(self, n)


class SpaceToDepthConv(nn.Module):
    """Stride-2 downsampling without discarding pixels: pixel-unshuffle(2) then 3x3 conv."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = ConvBNAct(cin * 4, cout, 3, 1)

    def forward(self, x):
        return self.conv(F.pixel_unshuffle(x, 2))


class ELAN(nn.Module):
    """CSP/ELAN aggregation: two 1x1 entries, a chain of RepConvs, concat all intermediate outputs, 1x1 out."""

    def __init__(self, cin, cout, n=2, hidden_ratio=0.5):
        super().__init__()
        h = max(8, int(cout * hidden_ratio))
        self.a = ConvBNAct(cin, h, 1)
        self.b = ConvBNAct(cin, h, 1)
        self.chain = nn.ModuleList([RepConv(h, h) for _ in range(n)])
        self.out = ConvBNAct(h * (2 + n), cout, 1)

    def forward(self, x):
        outs = [self.a(x), self.b(x)]
        y = outs[-1]
        for m in self.chain:
            y = m(y)
            outs.append(y)
        return self.out(torch.cat(outs, 1))


class LinearAttention(nn.Module):
    """ReLU linear attention (EfficientViT-style) with BN instead of LN; O(N) in tokens."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.proj = ConvBNAct(c, c, 1, act=False)
        self.ffn = nn.Sequential(ConvBNAct(c, c * 2, 1), ConvBNAct(c * 2, c, 1, act=False))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).reshape(B, 3, self.h, C // self.h, H * W).unbind(1)
        q, k = F.relu(q), F.relu(k)
        # scale k by 1/N BEFORE the token reduction: every partial sum stays bounded (FP16 / TensorRT safe);
        # the num/den ratio is unchanged. Computed in fp32 during mixed-precision training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = q.float(), k.float() / float(H * W), v.float()
            kv = torch.matmul(v, k.transpose(-1, -2))                                   # B,h,d,d
            num = torch.matmul(kv, q)                                                   # B,h,d,N
            den = torch.matmul(k.sum(-1, keepdim=True).transpose(-1, -2), q) + 1e-4    # B,h,1,N
            y = (num / den).reshape(B, C, H, W).to(x.dtype)
        x = x + self.proj(y)
        return x + self.ffn(x)


class SelectFuse(nn.Module):
    """Per-pixel scale selection: fused = sum_i softmax_i(w(x)) * F_i (STEERER-like selection, no concat)."""

    def __init__(self, c, n_in):
        super().__init__()
        self.w = nn.Conv2d(c * n_in, n_in, 1)
        self.refine = RepConv(c, c)

    def forward(self, feats):
        w = torch.softmax(self.w(torch.cat(feats, 1)), 1)
        y = sum(w[:, i:i + 1] * f for i, f in enumerate(feats))
        return self.refine(y)


# ----------------------------------------------------------------------------- network
@dataclass
class Cfg:
    widths: tuple = (24, 48, 96, 192, 256)   # stem, S2(4), S3(8), S4(16), S5(32)
    depths: tuple = (1, 2, 2, 1)             # RepConvs per ELAN in S2..S5
    neck: tuple = (32, 64, 96, 128)          # P2, P3, P4, P5 channels
    k8: int = 4                              # reference points per stride-8 cell
    topk: int = 8192                         # deploy cap on detections (2048 clipped a 2.4k-person Hajj photo)
    head_height: float = 1.65
    cam_sigma_deg: float = 1.0               # assumed pitch/roll error of the camera estimate (lower when locked)
    cam_h_relsd: float = 0.10                # relative camera-height error
    person_h_sd: float = 0.10                # head-height spread of people (m)


PRESETS = {
    "n": Cfg(widths=(16, 32, 64, 128, 192), depths=(1, 1, 2, 1), neck=(24, 48, 64, 96)),
    "s": Cfg(),
    "m": Cfg(widths=(32, 64, 128, 256, 384), depths=(1, 2, 3, 1), neck=(48, 96, 128, 192)),
    "l": Cfg(widths=(48, 96, 192, 384, 512), depths=(1, 3, 4, 2), neck=(64, 128, 192, 256)),
}


class Backbone(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        w, d = cfg.widths, cfg.depths
        self.stem = nn.Sequential(SpaceToDepthConv(3, w[0]), SpaceToDepthConv(w[0], w[1]))   # stride 4
        self.s2 = ELAN(w[1], w[1], d[0])
        self.s3 = nn.Sequential(RepConv(w[1], w[2], 2), ELAN(w[2], w[2], d[1]))
        self.s4 = nn.Sequential(RepConv(w[2], w[3], 2), ELAN(w[3], w[3], d[2]))
        self.s5 = nn.Sequential(RepConv(w[3], w[4], 2), ELAN(w[4], w[4], d[3]), LinearAttention(w[4]))

    def forward(self, x):
        c2 = self.s2(self.stem(x))
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        return c2, c3, c4, c5


class Neck(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        w, n = cfg.widths, cfg.neck
        self.l5 = ConvBNAct(w[4], n[3], 1)
        self.l4, self.u54 = ConvBNAct(w[3], n[2], 1), ConvBNAct(n[3], n[2], 1)
        self.l3, self.u43 = ConvBNAct(w[2], n[1], 1), ConvBNAct(n[2], n[1], 1)
        self.l2, self.u32 = ConvBNAct(w[1], n[0], 1), ConvBNAct(n[1], n[0], 1)
        self.f4, self.f3, self.f2 = SelectFuse(n[2], 2), SelectFuse(n[1], 2), SelectFuse(n[0], 2)

    @staticmethod
    def up(x):
        return F.interpolate(x, scale_factor=2.0, mode="nearest")

    def forward(self, c2, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.f4([self.l4(c4), self.up(self.u54(p5))])
        p3 = self.f3([self.l3(c3), self.up(self.u43(p4))])
        p2 = self.f2([self.l2(c2), self.up(self.u32(p3))])
        return p2, p3, p4, p5


class CodeNet(nn.Module):
    def __init__(self, cfg: Cfg | str = "s"):
        super().__init__()
        cfg = PRESETS[cfg] if isinstance(cfg, str) else cfg
        self.cfg = cfg
        self.deploy = False
        n = cfg.neck
        self.backbone = Backbone(cfg)
        self.neck = Neck(cfg)
        # point heads
        self.h8 = nn.Sequential(RepConv(n[1], n[1]), nn.Conv2d(n[1], cfg.k8 * 4, 1))   # per ref: logit, dx, dy, log(size/8)
        self.h4 = nn.Sequential(RepConv(n[0], n[0]), nn.Conv2d(n[0], 4, 1))     # logit, dx, dy, log(size/4)
        self.gate = nn.Sequential(RepConv(n[1], n[1] // 2), nn.Conv2d(n[1] // 2, 1, 1))  # P(dense) per s8 cell
        # per-person geometry at stride 8: (log metric distance camera->head, log Laplace scale)
        self.pgeo = nn.Sequential(RepConv(n[1], n[1] // 2), nn.Conv2d(n[1] // 2, 2, 1))
        # camera head from global S5 context
        c5 = cfg.widths[4]
        self.cam = nn.Sequential(nn.Linear(c5 * 2, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))
        with torch.no_grad():   # prior: FoV 60 deg, pitch 30 deg down, roll 0, camera height 6 m
            self.cam[-1].weight.mul_(0.01)
            self.cam[-1].bias.copy_(torch.tensor([math.log(math.radians(60)), math.radians(30), 0.0, math.log(6.0)]))
        # training-only auxiliary heads
        self.aux_depth = nn.Sequential(RepConv(n[1], n[1] // 2), nn.Conv2d(n[1] // 2, 1, 1))   # log depth
        self.aux_up = nn.Sequential(RepConv(n[1], n[1] // 2), nn.Conv2d(n[1] // 2, 2, 1))      # up-vector field
        self.aux_density = nn.Sequential(RepConv(n[0], n[0]), nn.Conv2d(n[0], 1, 1))           # density (stride 4)
        for h in [self.h8, self.h4]:
            nn.init.constant_(h[-1].bias, 0.0)
            with torch.no_grad():
                h[-1].bias[0::4].fill_(-4.6)     # score prior ~1%
                h[-1].bias[3::4].fill_(0.65)     # head size prior ~2x stride
                h[-1].weight.mul_(0.1)
        with torch.no_grad():
            self.pgeo[-1].weight.mul_(0.1)
            self.pgeo[-1].bias.copy_(torch.tensor([math.log(15.0), math.log(0.3)]))

    # ------------------------------------------------------------------ helpers
    def _anchors(self, H, W, stride, k, device):
        """Reference points (x, y) for every cell, k per cell on a sqrt(k) x sqrt(k) sub-grid."""
        r = int(round(math.sqrt(k)))
        sub = (torch.arange(r, device=device, dtype=torch.float32) + 0.5) / r
        gy, gx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
        sy, sx = torch.meshgrid(sub, sub, indexing="ij")
        ax = (gx[..., None] + sx.reshape(-1)) * stride       # H,W,k
        ay = (gy[..., None] + sy.reshape(-1)) * stride
        return torch.stack([ax, ay], -1)                     # H,W,k,2

    def camera(self, c5):
        g = torch.cat([F.adaptive_avg_pool2d(c5, 1), F.adaptive_max_pool2d(c5, 1)], 1).flatten(1)
        p = self.cam(g)
        return {"fov": p[:, 0].exp().clamp(0.2, 3.0), "pitch": p[:, 1], "roll": p[:, 2],
                "height": p[:, 3].exp().clamp(1.5, 400.0)}   # CCTV 2-15 m .. drones up to ~300 m

    def lift(self, uv, cam, img_hw, logd_net, logb_net, with_dist=False):
        """Pixel (u,v) -> gravity-aligned ray; distance = fusion of head-plane intersection and learned distance.

        Plane: d_p = drop / sin(a), a = ray angle below the horizon; d log d_p / d a = -1 / tan(a), so a 1 deg
        camera error is ~4 % at a = 45 deg but ~60 % at a = 2.6 deg (3 m camera, 30 m away) and undefined above the
        horizon. The learned distance has its own predicted Laplace scale b (log units, var = 2 b^2); the two are
        combined by inverse variance in log space. Returns BEV (X right, Z forward) metres, fused log-sigma, valid.
        """
        H, W = img_hw
        c = self.cfg
        f = (W * 0.5) / torch.tan(cam["fov"] * 0.5)                         # B
        x = (uv[..., 0] - W * 0.5) / f[:, None]
        y = (uv[..., 1] - H * 0.5) / f[:, None]
        z = torch.ones_like(x)
        cr, sr = torch.cos(cam["roll"])[:, None], torch.sin(cam["roll"])[:, None]
        x, y = cr * x - sr * y, sr * x + cr * y                              # undo roll
        cp, sp = torch.cos(cam["pitch"])[:, None], torch.sin(cam["pitch"])[:, None]
        yg, zg = cp * y + sp * z, -sp * y + cp * z                          # gravity-aligned (y down)
        r = torch.sqrt(x * x + yg * yg + zg * zg)
        sin_a = yg / r
        drop = (cam["height"] - c.head_height).clamp(min=0.1)[:, None]
        tan_a = sin_a / torch.sqrt((1 - sin_a * sin_a).clamp(min=1e-6))
        ok_p = sin_a > 1e-3
        logd_p = torch.log(drop / sin_a.clamp(min=1e-3))
        sig_ang = math.radians(c.cam_sigma_deg)
        var_p = (sig_ang / tan_a.clamp(min=1e-3)) ** 2 + c.cam_h_relsd ** 2 * (cam["height"][:, None] / drop) ** 2 \
            + (c.person_h_sd / drop) ** 2
        var_p = torch.where(ok_p, var_p, torch.full_like(var_p, 1e6))
        b = torch.exp(logb_net.clamp(-4, 3))
        var_n = 2 * b * b
        w_p, w_n = 1.0 / var_p, 1.0 / var_n
        logd = (w_p * torch.where(ok_p, logd_p, torch.zeros_like(logd_p)) + w_n * logd_net) / (w_p + w_n)
        sigma = torch.rsqrt(w_p + w_n)
        d = torch.exp(logd)
        X, Z = x / r * d, zg / r * d                                        # horizontal part of the unit ray * d
        valid = (X * X + Z * Z) > 1e-12      # any ground position (drones look straight down: heads behind the nadir point are valid)
        if with_dist:
            return torch.stack([X, Z], -1), sigma, valid, d
        return torch.stack([X, Z], -1), sigma, valid

    # ------------------------------------------------------------------ forward
    def forward(self, img):
        B, _, H, W = img.shape
        c2, c3, c4, c5 = self.backbone(img)
        p2, p3, _, _ = self.neck(c2, c3, c4, c5)
        k8 = self.cfg.k8
        o8 = self.h8(p3)                                     # B, k8*3, H/8, W/8
        o4 = self.h4(p2)                                     # B, 3,    H/4, W/4
        gate_logit = self.gate(p3)                           # B, 1,    H/8, W/8
        gate = torch.sigmoid(gate_logit)
        pg = self.pgeo(p3)                                   # B, 2,    H/8, W/8
        cam = self.camera(c5)

        h8, w8, h4, w4 = o8.shape[2], o8.shape[3], o4.shape[2], o4.shape[3]
        o8 = o8.reshape(B, k8, 4, h8, w8).permute(0, 3, 4, 1, 2)            # B,h8,w8,k8,4
        o4 = o4.permute(0, 2, 3, 1)                                         # B,h4,w4,4
        a8 = self._anchors(h8, w8, 8, k8, img.device)
        a4 = self._anchors(h4, w4, 4, 1, img.device)[:, :, 0]
        pts8 = a8 + o8[..., 1:3] * 8.0
        pts4 = a4 + o4[..., 1:3] * 4.0
        lsz8 = o8[..., 3] + math.log(8.0)                                   # log head size (input pixels)
        lsz4 = o4[..., 3] + math.log(4.0)
        if not self.deploy:
            return {"logit8": o8[..., 0], "pts8": pts8, "logit4": o4[..., 0], "pts4": pts4,
                    "gate": gate[:, 0], "gate_logit": gate_logit[:, 0], "pgeo": pg, "cam": cam,
                    "lsize8": lsz8, "lsize4": lsz4,
                    "aux_depth": self.aux_depth(p3)[:, 0], "aux_up": self.aux_up(p3),
                    "aux_density": F.relu(self.aux_density(p2))[:, 0]}

        # ---------------- deploy: static-shape decoding, all in graph
        g8 = (gate[:, 0] > 0.5).float()                                     # hard gate per s8 cell
        s8 = torch.sigmoid(o8[..., 0]) * (1.0 - g8)[..., None]
        s4 = torch.sigmoid(o4[..., 0])
        s4 = s4 * (s4 == F.max_pool2d(s4[:, None], 3, 1, 1)[:, 0]).float()  # local-max suppression
        s4 = s4 * F.interpolate(g8[:, None], size=(h4, w4), mode="nearest")[:, 0]
        pg8 = pg.permute(0, 2, 3, 1)                                        # B,h8,w8,2
        pg8k = pg8[:, :, :, None, :].expand(B, h8, w8, k8, 2)
        pg4 = F.interpolate(pg, size=(h4, w4), mode="nearest").permute(0, 2, 3, 1)
        scores = torch.cat([s8.reshape(B, -1), s4.reshape(B, -1)], 1)
        pts = torch.cat([pts8.reshape(B, -1, 2), pts4.reshape(B, -1, 2)], 1)
        lsz = torch.cat([lsz8.reshape(B, -1), lsz4.reshape(B, -1)], 1)
        geo = torch.cat([pg8k.reshape(B, -1, 2), pg4.reshape(B, -1, 2)], 1)
        top, idx = torch.topk(scores, self.cfg.topk, dim=1)
        uv = torch.gather(pts, 1, idx[..., None].expand(-1, -1, 2))
        gg = torch.gather(geo, 1, idx[..., None].expand(-1, -1, 2))
        size = torch.exp(torch.gather(lsz, 1, idx))
        bev, sigma, valid = self.lift(uv, cam, (H, W), gg[..., 0], gg[..., 1])
        cam_vec = torch.stack([cam["fov"], cam["pitch"], cam["roll"], cam["height"]], 1)
        return top, uv, size, bev, sigma, valid.float(), cam_vec

    def to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepConv) and m.fused is None:
                m.reparameterize()
        for n in ["aux_depth", "aux_up", "aux_density"]:
            delattr(self, n)
        self.deploy = True
        return self.eval()
