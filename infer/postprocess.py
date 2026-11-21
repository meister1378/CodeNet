"""BEV post-processing: camera smoothing, crowd clustering, metric polygons, image overlay polygons.

Clustering (DBSCAN with an uncertainty-aware metric)
    Each person has a BEV position (X right, Z forward, metres) and a fused log-distance sigma. The error of a
    monocular position is mostly along the viewing ray, so the separation of two people is measured as
        m = sqrt(t^2 + (r / s)^2),   s = clip(rad_k * d * sigma, 1, rad_max)   (rms of the pair)
    with t / r the lateral / radial components. A person with >= min_nb neighbours within eps is a core person;
    clusters grow only through core people, so a single person in between does not chain two groups.
    Clusters wider than split_extent are split at density valleys: on a smoothed BEV density map every peak
    owns a basin; neighbouring basins stay together unless the density between them drops below split_valley x
    the lower peak (a uniformly dense crowd stays one cluster, two groups joined by a thin band are separated).
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
    min_nb: int = 3           # neighbours needed for a core person
    rad_k: float = 0.5        # radial tolerance per unit of (distance * sigma)
    rad_max: float = 2.0      # cap on the radial tolerance factor (far people)
    min_people: int = 5       # smallest reported cluster
    split_extent: float = 10.0  # clusters wider than this (m) are checked for density valleys
    split_valley: float = 0.7  # merge two density peaks if the valley between them is >= this x the lower peak
    split_sigma: float = 1.5  # smoothing of the BEV density map (m) - the scale of the parts
    split_cell: float = 0.5   # density map cell (m)
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


def _dbscan(bev, d, s, eps, min_nb, device):
    """Raw DBSCAN labels (-1 = noise) with the uncertainty-aware metric."""
    n = len(bev)
    lab = -np.ones(n, np.int64)
    if n < 2:
        return lab
    if device is not None and str(device).startswith("cuda"):
        ii, jj = _edges_torch(bev, d, s, eps, device)
    else:
        ii, jj = _edges_kdtree(bev, d, s, eps)
    deg = np.bincount(ii, minlength=n) + np.bincount(jj, minlength=n)
    core = deg >= min_nb
    cc = core[ii] & core[jj]
    if core.any():
        g = coo_matrix((np.ones(cc.sum()), (ii[cc], jj[cc])), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        lab[core] = comp[core]
        for a, b in ((ii, jj), (jj, ii)):            # border people join the cluster of a core neighbour
            m = core[a] & ~core[b] & (lab[b] < 0)
            lab[b[m]] = lab[a[m]]
    return lab


def _density_split(idx, bev, cfg):
    """Split one connected crowd at density valleys (mode seeking on a smoothed BEV density map).

    People are splatted on a cell grid and smoothed with a Gaussian (split_sigma m). Every cell climbs to its
    highest 8-neighbour until a local maximum, which gives one basin per density peak. Two touching basins are
    merged while the density on their shared border (saddle) is >= split_valley x the lower of the two peaks,
    i.e. a part is kept separate only where the crowd density drops by more than (1 - split_valley) between them.
    Parts smaller than min_people are merged into the neighbour with the highest saddle."""
    P = bev[idx]
    c = cfg.split_cell
    pad = 3 * cfg.split_sigma
    o = P.min(0) - pad
    gx = ((P[:, 0] - o[0]) / c).astype(np.int64)
    gz = ((P[:, 1] - o[1]) / c).astype(np.int64)
    H, W = int(gz.max() + pad / c) + 2, int(gx.max() + pad / c) + 2
    D = np.zeros((H, W), np.float32)
    np.add.at(D, (gz, gx), 1.0)
    D = cv2.GaussianBlur(D, (0, 0), cfg.split_sigma / c)
    # steepest-ascent pointer of every cell (8-neighbourhood), then pointer jumping to the peak
    Dp = np.pad(D, 1, constant_values=-1.0)
    best = np.arange(H * W).reshape(H, W)
    bestv = D.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            v = Dp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            m = v > bestv
            bestv = np.where(m, v, bestv)
            best = np.where(m, (yy + dy).clip(0, H - 1) * W + (xx + dx).clip(0, W - 1), best)
    ptr = best.reshape(-1)
    for _ in range(64):
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    basin = ptr.reshape(H, W)
    occ = D > 0.05 * D.max()
    peaks = np.unique(np.concatenate([basin[occ], basin[gz, gx]]))
    if len(peaks) < 2:
        return [idx]
    pk = {int(q): float(D.flat[q]) for q in peaks}
    # saddles between touching basins (horizontal + vertical neighbours inside the occupied region)
    sad = {}
    for a, b, va, vb, oa, ob in ((basin[:, :-1], basin[:, 1:], D[:, :-1], D[:, 1:], occ[:, :-1], occ[:, 1:]),
                                 (basin[:-1], basin[1:], D[:-1], D[1:], occ[:-1], occ[1:])):
        m = (a != b) & oa & ob
        for p, q, v in zip(a[m], b[m], np.minimum(va, vb)[m]):
            key = (min(p, q), max(p, q))
            if v > sad.get(key, -1.0):
                sad[key] = float(v)
    parent = {q: q for q in pk}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for (p, q), v in sorted(sad.items(), key=lambda kv: -kv[1]):
        rp, rq = find(p), find(q)
        if rp != rq and v >= cfg.split_valley * min(pk[rp], pk[rq]):
            hi, lo = (rp, rq) if pk[rp] >= pk[rq] else (rq, rp)
            parent[lo] = hi
    lab = np.array([find(int(q)) for q in basin[gz, gx]])
    # small parts -> neighbour with the highest saddle
    while True:
        ids, cnt = np.unique(lab, return_counts=True)
        small = ids[cnt < cfg.min_people]
        if len(small) == 0 or len(ids) == 1:
            break
        q = int(small[0])
        cand = [(v, find(b if find(a) == q else a)) for (a, b), v in sad.items() if q in (find(a), find(b)) and find(a) != find(b)]
        if not cand:
            nn = cKDTree(P[lab != q]).query(P[lab == q])[1]
            lab[lab == q] = lab[lab != q][nn]
        else:
            tgt = max(cand)[1]
            parent[q] = tgt
            lab[lab == q] = tgt
    return [idx[lab == q] for q in np.unique(lab)]


def cluster_people(bev: np.ndarray, sigma: np.ndarray, cfg: ClusterCfg = ClusterCfg(), device=None) -> np.ndarray:
    """Cluster label per person (-1 = not in a cluster of >= min_people). device: torch CUDA device for the
    pair test (fast for thousands of people); None = CPU KD-tree."""
    n = len(bev)
    lab = -np.ones(n, np.int64)
    if n < max(cfg.min_people, 2):
        return lab
    d = np.hypot(bev[:, 0], bev[:, 1])
    s = np.clip(cfg.rad_k * d * sigma, 1.0, cfg.rad_max)
    raw = _dbscan(bev, d, s, cfg.eps, cfg.min_nb, device)
    groups = []
    for c in np.unique(raw[raw >= 0]):
        idx = np.flatnonzero(raw == c)
        if len(idx) >= cfg.min_people:
            big = cfg.split_valley < 1.0 and np.ptp(bev[idx], 0).max() > cfg.split_extent
            groups += _density_split(idx, bev, cfg) if big else [idx]
    groups.sort(key=lambda g: -len(g))
    for k, g in enumerate(groups):
        lab[g] = k
    return lab


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
