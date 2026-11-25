"""BEV post-processing: camera smoothing and crowd zones from a ground-plane density field.

Density field
    Every person's BEV position (X right, Z forward, metres) is splatted on a ground grid (`cell` m) and blurred
    with a Gaussian of `bw` m, giving the local density F (people / m^2). For a static camera F is an exponential
    moving average over frames (`alpha`), so zones and levels change slowly; otherwise it is the frame's field.

Zones
    Cell state: 0 none, 1 crowd (F >= crowd), 2 caution (F >= risk[0]), 3 warning (F >= risk[1]). With the moving
    average a state drops only below (1 - hyst) x its threshold. Patches of a state smaller than `min_hot` m^2 are
    demoted. A zone is a connected region of state >= 1 holding >= min_people; its level is the highest state in it
    (safe / caution / warning). A continuous crowd stays one zone: dense spots inside it are coloured, not cut out.

Image regions
    Discs of `margin` metres (at each person's distance) around the detected head pixels of a zone (heads of the
    last `pool` frames for a static camera), closed at the same scale, on one down-scaled raster; overlaps go to the
    zone of the nearest head. Drawn from the pixels, so the outline stays on the people even where the metric
    position is off. Caution / warning patches are drawn the same way from the heads inside them.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

HEAD_HEIGHT = 1.65


@dataclass
class ZoneCfg:
    bw: float = 1.0           # kernel std (m): the scale of "local" density
    cell: float = 0.25        # ground grid cell (m)
    crowd: float = 1.0        # people / m^2 for a crowd zone
    risk: tuple = (2.0, 4.0)  # people / m^2 for caution / warning
    hyst: float = 0.25        # static camera: a state drops only below (1 - hyst) x its threshold
    alpha: float = 0.05       # static camera: moving average of the field per frame
    min_people: float = 5.0   # smallest zone (people)
    min_hot: float = 2.0      # smallest patch of a state (m^2)
    pool: int = 4             # static camera: frames of heads used for the image regions
    margin: float = 0.8       # image region margin around heads (m)
    img_ds: int = 4           # min down-scaling of the image raster (>= long side / 480)


@dataclass
class Patch:
    level: int                                      # 1 caution, 2 warning
    bev_polys: list = field(default_factory=list)   # (K,2) arrays, metres
    img_polys: list = field(default_factory=list)   # (K,2) int arrays, pixels
    geo: np.ndarray | None = None                   # pooled heads the image region is drawn from


@dataclass
class Zone:
    members: np.ndarray                             # indices into the frame's people
    area: float                                     # m^2
    people: float                                   # integral of the density field over the zone
    rho: float                                      # mean density (people / m^2)
    peak: float                                     # highest local density (people / m^2)
    level: int                                      # 0 safe, 1 caution, 2 warning
    bev_polys: list = field(default_factory=list)
    img_polys: list = field(default_factory=list)
    hot: list = field(default_factory=list)         # Patch list (caution / warning areas inside the zone)
    geo: np.ndarray | None = None

    @property
    def count(self) -> int:
        return len(self.members)


class CameraSmoother:
    """Running median of the per-frame camera estimate (fov, pitch, roll, height) for a fixed camera."""

    def __init__(self, window: int = 25):
        self.buf = deque(maxlen=window)

    def __call__(self, cam_vec: np.ndarray) -> np.ndarray:
        v = np.asarray(cam_vec, np.float64).copy()
        v[3] = math.log(v[3])                           # median of log height
        self.buf.append(v)
        m = np.median(np.stack(self.buf), 0)
        m[3] = math.exp(m[3])
        return m


def world_to_pixel(X, Y, Z, cam, img_wh, net_wh):
    """Inverse of CodeNet.lift: gravity frame (X right, Y down from the camera, Z forward) -> image pixel."""
    fov, pitch, roll = float(cam[0]), float(cam[1]), float(cam[2])
    W, H = img_wh
    nw, nh = net_wh
    f = (nw * 0.5) / math.tan(fov * 0.5)
    cp, sp = math.cos(pitch), math.sin(pitch)
    y = cp * Y - sp * Z
    z = sp * Y + cp * Z
    zs = np.maximum(z, 1e-6)
    xr, yr = X / zs, y / zs
    cr, sr = math.cos(roll), math.sin(roll)
    xn, yn = cr * xr + sr * yr, -sr * xr + cr * yr
    return np.stack([(xn * f + nw * 0.5) * W / nw, (yn * f + nh * 0.5) * H / nh], -1), z > 0.05


def _contours(mask, cell, origin):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(q, 0.75, True)[:, 0].astype(np.float64) * cell + origin for q in cs if len(q) >= 3]


class DensityZones:
    """Crowd zones of one camera. static=True: moving average + hysteresis over consecutive frames (call reset()
    between unrelated videos); static=False: every call is independent (images, moving cameras)."""

    def __init__(self, cfg: ZoneCfg = ZoneCfg(), static: bool = False):
        self.c, self.static = cfg, static
        self.reset()

    def reset(self):
        self.F = self.S = self.o = None
        self.frames = 0
        self.heads = deque(maxlen=self.c.pool if self.static else 1)

    def _grid(self, bev):
        """Ground grid covering the people; for a static camera it only grows (old field copied in)."""
        c = self.c
        lo, hi = bev.min(0) - 6 * c.bw, bev.max(0) + 6 * c.bw
        keep = self.F is not None and self.static
        if keep:
            (h, w), o = self.F.shape, self.o
            if (lo >= o).all() and (hi <= o + np.array([w, h]) * c.cell).all():
                return
            lo, hi = np.minimum(lo - 10, o), np.maximum(hi + 10, o + np.array([w, h]) * c.cell)
        o = np.floor(lo / c.cell) * c.cell
        w, h = np.ceil((hi - o) / c.cell).astype(int) + 1
        F, S = np.zeros((h, w), np.float32), np.zeros((h, w), np.int8)
        if keep:
            dx, dy = np.round((self.o - o) / c.cell).astype(int)
            oh, ow = self.F.shape
            F[dy:dy + oh, dx:dx + ow], S[dy:dy + oh, dx:dx + ow] = self.F, self.S
        self.F, self.S, self.o = F, S, o

    def _field(self, bev):
        """Bilinear splat of the people + Gaussian blur -> people / m^2."""
        c = self.c
        h, w = self.F.shape
        g = (bev - self.o) / c.cell
        x0, y0 = np.floor(g[:, 0]).astype(int), np.floor(g[:, 1]).astype(int)
        fx, fy = g[:, 0] - x0, g[:, 1] - y0
        A = np.zeros((h, w), np.float32)
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(A, (np.clip(y0 + dy, 0, h - 1), np.clip(x0 + dx, 0, w - 1)), wt)
        return cv2.GaussianBlur(A, (0, 0), c.bw / c.cell) / (c.cell * c.cell)

    def _states(self, Fn):
        c = self.c
        th = np.array([c.crowd, *c.risk], np.float32)
        if self.static and self.frames > 0:
            self.F += c.alpha * (Fn - self.F)
            up = (self.F[..., None] >= th).sum(-1).astype(np.int8)
            dn = (self.F[..., None] >= th * (1 - c.hyst)).sum(-1).astype(np.int8)
            S = np.maximum(up, np.minimum(self.S, dn))
        else:
            self.F = Fn
            S = (Fn[..., None] >= th).sum(-1).astype(np.int8)
        min_cells = c.min_hot / c.cell ** 2
        for L in (3, 2, 1):                                  # demote small patches, top level first
            _, lab, st, _ = cv2.connectedComponentsWithStats((S >= L).astype(np.uint8), connectivity=8)
            small = np.flatnonzero(st[:, cv2.CC_STAT_AREA] < min_cells)
            small = small[small > 0]
            if len(small):
                S[np.isin(lab, small) & (S >= L)] = L - 1
        nz, zl = cv2.connectedComponents((S >= 1).astype(np.uint8), connectivity=8)
        people = np.bincount(zl.ravel(), weights=self.F.ravel(), minlength=nz) * c.cell ** 2
        S[np.isin(zl, np.flatnonzero(people < c.min_people)) & (zl > 0)] = 0
        self.S = S

    def _lookup(self, P, grid):
        g = np.floor((P - self.o) / self.c.cell).astype(int)
        h, w = grid.shape
        ok = (g[:, 0] >= 0) & (g[:, 0] < w) & (g[:, 1] >= 0) & (g[:, 1] < h)
        g = np.clip(g, 0, [w - 1, h - 1])
        return np.where(ok, grid[g[:, 1], g[:, 0]], 0)

    def __call__(self, bev, uv, cam, img_wh):
        """-> labels (zone index per person, -1 outside), zones (most people first)."""
        c = self.c
        if not self.static:
            self.reset()
        bev = np.asarray(bev, np.float64).reshape(-1, 2)
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        n = len(bev)
        if n:
            self._grid(bev)
        if self.F is None:
            return -np.ones(0, np.int64), []
        self._states(self._field(bev) if n else np.zeros_like(self.F))
        self.frames += 1
        S, F = self.S, self.F
        nz, zl = cv2.connectedComponents((S >= 1).astype(np.uint8), connectivity=8)
        people = np.bincount(zl.ravel(), weights=F.ravel(), minlength=nz) * c.cell ** 2
        area = np.bincount(zl.ravel(), minlength=nz) * c.cell ** 2
        lvl = np.zeros(nz, int)
        np.maximum.at(lvl, zl.ravel(), S.ravel())
        peak = np.zeros(nz, np.float32)
        np.maximum.at(peak, zl.ravel(), F.ravel())

        self.heads.appendleft((bev, uv))
        PB = np.concatenate([b for b, _ in self.heads]).reshape(-1, 2)
        PU = np.concatenate([u for _, u in self.heads]).reshape(-1, 2)
        zp = self._lookup(bev, zl) if n else np.zeros(0, int)
        hz, hs = self._lookup(PB, zl), self._lookup(PB, S)
        labels = -np.ones(n, np.int64)
        zones = []
        for k, z in enumerate(sorted(range(1, nz), key=lambda z: -people[z])):
            labels[zp == z] = k
            m = zl == z
            zn = Zone(members=np.flatnonzero(zp == z), area=float(area[z]), people=float(people[z]),
                      rho=float(people[z] / max(area[z], 1e-6)), peak=float(peak[z]), level=int(lvl[z]) - 1,
                      bev_polys=_contours(m, c.cell, self.o), geo=np.flatnonzero(hz == z))
            for L in (2, 3):
                if (m & (S >= L)).any():
                    zn.hot.append(Patch(L - 1, _contours(m & (S >= L), c.cell, self.o),
                                        geo=np.flatnonzero((hz == z) & (hs >= L))))
            zones.append(zn)

        W, H = img_wh
        f_img = (W * 0.5) / math.tan(float(cam[0]) * 0.5)
        drop = max(float(cam[3]) - HEAD_HEIGHT, 0.1)
        zs = [z for z in zones if len(z.geo)]
        _image_polygons(zs, PU, PB, f_img, drop, (W, H), c)
        for L in (1, 2):
            _image_polygons([p for z in zs for p in z.hot if p.level == L and len(p.geo)], PU, PB, f_img, drop,
                            (W, H), c)
        return labels, zones


def _disc_mask(pts, radius, origin, cell, shape):
    m = np.zeros(shape, np.uint8)
    c = np.round((pts - origin) / cell).astype(np.int32)
    r = np.maximum(1, np.round(np.broadcast_to(radius, len(pts)) / cell).astype(np.int32))
    for x, y, rr in np.unique(np.concatenate([c, r[:, None]], 1), axis=0):
        cv2.circle(m, (int(x), int(y)), int(rr), 1, -1)
    return m


def _image_polygons(regions, uv, bev, f_img, drop, img_wh, cfg):
    """Image regions that do not overlap: discs of `margin` metres (at each person's distance) around the head
    pixels of each region (`geo`), closed at the same scale, on one down-scaled raster; where regions overlap,
    each raster cell goes to the region of the nearest head."""
    if not regions:
        return
    W, H = img_wh
    ds = max(cfg.img_ds, int(round(max(W, H) / 480)))
    gh, gw = H // ds + 1, W // ds + 1
    own = np.full((gh, gw), -1, np.int32)
    nown = np.zeros((gh, gw), np.uint8)
    regs = []
    for k, c in enumerate(regions):
        U, P = uv[c.geo], bev[c.geo]
        rpx = np.clip(f_img * cfg.margin / np.sqrt((P ** 2).sum(1) + drop ** 2), 3.0, 0.25 * max(W, H))
        kc = max(1, int(round(float(np.median(rpx)) / ds)))
        pad = int(math.ceil(rpx.max() / ds)) + kc + 2
        x0, y0 = (np.floor(U.min(0) / ds) - pad).astype(int).clip(0, [gw - 1, gh - 1])
        x1, y1 = (np.ceil(U.max(0) / ds) + pad).astype(int).clip(0, [gw - 1, gh - 1])
        roi = _disc_mask(U, rpx, np.array([x0, y0]) * ds, ds, (y1 - y0 + 1, x1 - x0 + 1))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kc + 1,) * 2))
        m = np.zeros((gh, gw), np.uint8)
        m[y0:y1 + 1, x0:x1 + 1] = roi
        regs.append(m)
        own[m > 0] = k
        nown += m
    over = nown > 1
    if over.any():
        idx = np.concatenate([c.geo for c in regions])
        lab = np.concatenate([np.full(len(c.geo), k) for k, c in enumerate(regions)])
        yy, xx = np.nonzero(over)
        _, nn = cKDTree(uv[idx] / ds).query(np.stack([xx, yy], 1))
        own[yy, xx] = lab[nn]
    for k, c in enumerate(regions):
        m = ((own == k) & (regs[k] > 0)).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c.img_polys = [np.round(cv2.approxPolyDP(q, 0.6, True)[:, 0] * ds).astype(np.int32) for q in cs
                       if len(q) >= 3 and cv2.contourArea(q) >= 2]
