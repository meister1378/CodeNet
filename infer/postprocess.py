"""BEV post-processing: camera smoothing, crowd clustering, metric polygons, image overlay polygons.

Clustering (DBSCAN with an uncertainty-aware metric)
    Each person has a BEV position (X right, Z forward, metres) and a fused log-distance sigma. The error of a
    monocular position is mostly along the viewing ray, so the separation of two people is measured as
        m = sqrt(t^2 + (r / s)^2),   s = max(1, rad_k * d * sigma)   (rms of the pair)
    with t / r the lateral / radial components. A person with >= min_nb neighbours within eps is a core person;
    clusters grow only through core people, so a single person in between does not chain two groups.
    The pair test runs on the GPU in row chunks when a CUDA device is given (thousands of people in ~1 ms);
    on the CPU, candidates come from a KD-tree with a per-person radius eps * sqrt(1 + s_i^2), which contains
    every pair with m <= eps, and the exact metric is evaluated on those candidates only.

Polygons
    BEV: union of discs (margin metres) around the members, closed by `close` metres, rasterised on a small
    per-cluster grid (cell metres). The area of that region is the occupied area used for the density.
    Image: the BEV outline projected at head height, united with discs of `margin` metres (at each person's
    distance) around the detected head pixels, so every member lies inside the drawn polygon even where the
    monocular BEV position and the pixel disagree by a few pixels. Rasterised per cluster on a down-scaled ROI.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HEAD_HEIGHT = 1.65


@dataclass
class ClusterCfg:
    eps: float = 1.5          # neighbour distance (m)
    min_nb: int = 2           # neighbours needed for a core person
    rad_k: float = 0.5        # radial tolerance per unit of (distance * sigma)
    min_people: int = 5       # smallest reported cluster
    margin: float = 0.8       # polygon margin around people (m)
    close: float = 0.8        # morphological closing of the BEV region (m)
    cell: float = 0.2         # BEV raster cell (m)
    img_ds: int = 4           # min down-scaling of the image ROI raster (>= long side / 480)


@dataclass
class Cluster:
    members: np.ndarray                    # indices into the frame's people
    area: float                            # occupied area (m^2)
    bev_polys: list = field(default_factory=list)   # list of (K,2) arrays, metres
    img_polys: list = field(default_factory=list)   # list of (K,2) int arrays, pixels

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def density(self) -> float:
        return self.count / max(self.area, 1e-6)


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


def _pair_ok(Di, Dj_minus, ui, uj, si, sj, eps):
    """Exact uncertainty-aware separation test for candidate pairs (numpy or torch arrays)."""
    um = ui + uj
    nrm = (um * um).sum(-1, keepdims=True) ** 0.5
    um = um / (nrm + 1e-6)
    rad = (Dj_minus * um).sum(-1)
    tan2 = (Dj_minus * Dj_minus).sum(-1) - rad * rad
    return tan2 + rad * rad / (0.5 * (si * si + sj * sj)) <= eps * eps


def _edges_kdtree(bev, d, s, eps):
    """CPU: KD-tree candidates (radius eps * sqrt(1 + s_i^2) holds every pair with m <= eps), exact test."""
    n = len(bev)
    nbrs = cKDTree(bev).query_ball_point(bev, eps * np.sqrt(1.0 + s * s))
    ii = np.repeat(np.arange(n), [len(v) for v in nbrs])
    jj = np.concatenate([np.asarray(v, np.int64) for v in nbrs])
    keep = ii < jj
    ii, jj = ii[keep], jj[keep]
    u = bev / np.maximum(d, 1e-6)[:, None]
    ok = _pair_ok(None, bev[ii] - bev[jj], u[ii], u[jj], s[ii], s[jj], eps)
    return ii[ok], jj[ok]


def _edges_torch(bev, d, s, eps, device, chunk=1024):
    """GPU: exact test on all pairs, row chunks (N x N never materialised for large N)."""
    import torch
    B = torch.as_tensor(bev, dtype=torch.float32, device=device)
    S = torch.as_tensor(s, dtype=torch.float32, device=device)
    U = B / torch.as_tensor(d, dtype=torch.float32, device=device).clamp(min=1e-6)[:, None]
    n, out_i, out_j = len(bev), [], []
    col = torch.arange(n, device=device)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        ok = _pair_ok(None, B[i0:i1, None] - B[None], U[i0:i1, None], U[None], S[i0:i1, None], S[None], eps)
        ok &= col[None] > torch.arange(i0, i1, device=device)[:, None]
        a, b = ok.nonzero(as_tuple=True)
        out_i.append(a + i0)
        out_j.append(b)
    return torch.cat(out_i).cpu().numpy(), torch.cat(out_j).cpu().numpy()


def cluster_people(bev: np.ndarray, sigma: np.ndarray, cfg: ClusterCfg = ClusterCfg(), device=None) -> np.ndarray:
    """Cluster label per person (-1 = not in a cluster of >= min_people). device: torch CUDA device for the
    pair test (fast for thousands of people); None = CPU KD-tree."""
    n = len(bev)
    lab = -np.ones(n, np.int64)
    if n < max(cfg.min_people, 2):
        return lab
    d = np.hypot(bev[:, 0], bev[:, 1])
    s = np.maximum(1.0, cfg.rad_k * d * sigma)
    if device is not None and str(device).startswith("cuda"):
        ii, jj = _edges_torch(bev, d, s, cfg.eps, device)
    else:
        ii, jj = _edges_kdtree(bev, d, s, cfg.eps)
    deg = np.bincount(ii, minlength=n) + np.bincount(jj, minlength=n)
    core = deg >= cfg.min_nb
    cc = core[ii] & core[jj]
    if core.any():
        g = coo_matrix((np.ones(cc.sum()), (ii[cc], jj[cc])), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        lab[core] = comp[core]
        # border people join the cluster of a core neighbour
        for a, b in ((ii, jj), (jj, ii)):
            m = core[a] & ~core[b] & (lab[b] < 0)
            lab[b[m]] = lab[a[m]]
    # drop small clusters, renumber by size
    ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    big = ids[cnt >= cfg.min_people][np.argsort(-cnt[cnt >= cfg.min_people], kind="stable")]
    remap = -np.ones(lab.max() + 2 if len(lab) else 1, np.int64)
    remap[big] = np.arange(len(big))
    return np.where(lab >= 0, remap[lab], -1)


def _disc_mask(pts, radius, origin, cell, shape):
    m = np.zeros(shape, np.uint8)
    c = np.round((pts - origin) / cell).astype(np.int32)
    r = np.maximum(1, np.round(np.broadcast_to(radius, len(pts)) / cell).astype(np.int32))
    for (x, y), rr in zip(c, r):
        cv2.circle(m, (int(x), int(y)), int(rr), 1, -1)
    return m


def build_clusters(bev, sigma, uv, cam, img_wh, net_wh, cfg: ClusterCfg = ClusterCfg(), device=None):
    """Cluster labels + per-cluster occupied area, BEV outline and image overlay polygon."""
    labels = cluster_people(bev, sigma, cfg, device)
    W, H = img_wh
    out = []
    f_img = (W * 0.5) / math.tan(float(cam[0]) * 0.5)
    drop = max(float(cam[3]) - HEAD_HEIGHT, 0.1)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(cfg.close / cfg.cell) + 1,) * 2)
    for c in range(labels.max() + 1 if len(labels) else 0):
        mem = np.flatnonzero(labels == c)
        P = bev[mem]
        # ---- BEV region
        pad = cfg.margin + cfg.close + cfg.cell
        o = P.min(0) - pad
        shape = (int(math.ceil((P[:, 1].max() + pad - o[1]) / cfg.cell)) + 1,
                 int(math.ceil((P[:, 0].max() + pad - o[0]) / cfg.cell)) + 1)
        m = cv2.morphologyEx(_disc_mask(P, cfg.margin, o, cfg.cell, shape), cv2.MORPH_CLOSE, k_close)
        area = float(m.sum()) * cfg.cell * cfg.cell
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bev_polys = [cv2.approxPolyDP(q, 0.75, True)[:, 0].astype(np.float64) * cfg.cell + o for q in cs if len(q) >= 3]
        # ---- image overlay: projected BEV outline (head height) U discs around the detected heads
        U = uv[mem]
        dist3 = np.sqrt((P ** 2).sum(1) + drop ** 2)
        rpx = np.clip(f_img * cfg.margin / dist3, 3.0, 0.25 * max(W, H))
        proj = []
        for q in bev_polys:                             # a pinhole maps straight plane edges to straight lines
            pix, ok = world_to_pixel(q[:, 0], np.full(len(q), drop), q[:, 1], cam, img_wh, net_wh)
            if ok.all():
                proj.append(pix)
        lo = np.floor(np.min([U.min(0) - rpx.max()] + [p.min(0) for p in proj], 0)).clip(0, [W - 1, H - 1])
        hi = np.ceil(np.max([U.max(0) + rpx.max()] + [p.max(0) for p in proj], 0)).clip(0, [W - 1, H - 1])
        ds = max(cfg.img_ds, int(round(max(W, H) / 480)))
        sh = (int((hi[1] - lo[1]) / ds) + 2, int((hi[0] - lo[0]) / ds) + 2)
        mi = _disc_mask(U, rpx, lo, ds, sh)
        for p in proj:
            cv2.fillPoly(mi, [np.round((np.clip(p, lo, hi) - lo) / ds).astype(np.int32)], 1)
        kc = max(1, int(round(float(np.median(rpx)) / ds)))
        mi = cv2.morphologyEx(mi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kc + 1,) * 2))
        cs, _ = cv2.findContours(mi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_polys = [np.round(cv2.approxPolyDP(q, 0.6, True)[:, 0] * ds + lo).astype(np.int32) for q in cs if len(q) >= 3]
        out.append(Cluster(members=mem, area=area, bev_polys=bev_polys, img_polys=img_polys))
    return labels, out
