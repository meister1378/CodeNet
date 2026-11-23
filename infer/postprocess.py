"""BEV post-processing: camera smoothing, crowd clustering, metric polygons, image overlay polygons.

Clustering (DBSCAN with an uncertainty-aware metric)
    Each person has a BEV position (X right, Z forward, metres) and a fused log-distance sigma. The error of a
    monocular position is mostly along the viewing ray, so the separation of two people is measured as
        m = sqrt(t^2 + (r / s)^2),   s = clip(rad_k * d * sigma, 1, rad_max)   (rms of the pair)
    with t / r the lateral / radial components. A person with >= min_nb neighbours within eps is a core person;
    clusters grow only through core people, so a single person in between does not chain two groups.
    Clusters wider than split_extent are split at density valleys: on a smoothed BEV density map every peak
    owns a basin; neighbouring basins stay together unless the density between them drops below split_valley x
    the lower peak (two groups joined by a thin band are separated). A uniformly dense crowd that is still wider
    than max_extent is then partitioned into compact neighbouring groups of about group_area m^2 (k-means on the
    ground plane), so area and density are reported per local group rather than for a whole plaza.
    The pair test runs on the GPU in row chunks when a CUDA device is given (thousands of people in ~1 ms);
    on the CPU, candidates come from a KD-tree with a per-person radius eps * sqrt(1 + s_i^2), which contains
    every pair with m <= eps, and the exact metric is evaluated on those candidates only.

Polygons
    BEV: union of discs (margin metres) around the members, closed by `close` metres, rasterised on a small
    per-cluster grid (cell metres). The area of that region is the occupied area used for the density.
    Image: discs of `margin` metres (at each person's distance) around the detected head pixels, closed at the
    same scale, on one down-scaled raster; overlaps between clusters are split by the nearest head. Built from the pixels (not from the projected BEV
    outline), so the outline stays on the people even where the metric position is off.
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
    max_extent: float = 12.0  # crowds still wider than this (m) are partitioned into compact groups (0 = off)
    group_area: float = 80.0  # target occupied area of one such group (m^2)
    margin: float = 0.8       # polygon margin around people (m)
    close: float = 0.8        # morphological closing of the BEV region (m)
    cell: float = 0.2         # BEV raster cell (m)
    img_ds: int = 4           # min down-scaling of the image ROI raster (>= long side / 480)
    risk: tuple = (2.0, 4.0)  # density (people / m^2) for caution / warning; below = safe


@dataclass
class Cluster:
    members: np.ndarray                    # indices into the frame's people
    area: float                            # occupied area (m^2)
    bev_polys: list = field(default_factory=list)   # list of (K,2) arrays, metres
    img_polys: list = field(default_factory=list)   # list of (K,2) int arrays, pixels
    geo: np.ndarray | None = None          # indices of the points the region is drawn from (default: members)
    tid: int = -1                          # track id (video tracking), -1 = untracked
    rho: float = 0.0                       # reported density (time-smoothed when tracked)
    level: int = 0                         # 0 safe, 1 caution, 2 warning

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def density(self) -> float:
        return self.count / max(self.area, 1e-6)


def risk_level(rho: float, risk, prev: int | None = None, hyst: float = 0.0) -> int:
    """Level from density; with a previous level, stepping down needs rho < (1 - hyst) x the threshold."""
    lv = int(rho >= risk[0]) + int(rho >= risk[1])
    if prev is not None and lv < prev:
        lv = max(lv, int(rho >= risk[0] * (1 - hyst)) + int(rho >= risk[1] * (1 - hyst)))
        lv = min(lv, prev)
    return lv


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


def _dbscan(bev, d, s, eps, min_nb, device, w=None, relax=None):
    """Raw DBSCAN labels (-1 = noise) with the uncertainty-aware metric.
    w: optional point weights (points pooled over K frames carry 1/K): a point is core when the weight within
    eps, itself included, is >= min_nb + 1 (= on average >= min_nb neighbours per frame).
    relax: optional per-point reduction of that requirement (hysteresis for people already in a cluster)."""
    n = len(bev)
    lab = -np.ones(n, np.int64)
    if n < 2:
        return lab
    if device is not None and str(device).startswith("cuda"):
        ii, jj = _edges_torch(bev, d, s, eps, device)
    else:
        ii, jj = _edges_kdtree(bev, d, s, eps)
    if w is None:
        deg = np.bincount(ii, minlength=n) + np.bincount(jj, minlength=n)
        core = deg >= min_nb
    else:
        deg = np.bincount(ii, weights=w[jj], minlength=n) + np.bincount(jj, weights=w[ii], minlength=n) + w
        core = deg >= min_nb + 1 - (0.0 if relax is None else relax) - 1e-6
    cc = core[ii] & core[jj]
    if core.any():
        g = coo_matrix((np.ones(cc.sum()), (ii[cc], jj[cc])), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        lab[core] = comp[core]
        for a, b in ((ii, jj), (jj, ii)):            # border people join the cluster of a core neighbour
            m = core[a] & ~core[b] & (lab[b] < 0)
            lab[b[m]] = lab[a[m]]
    return lab


def _density_split(idx, bev, cfg, w=None):
    """Split one connected crowd at density valleys (mode seeking on a smoothed BEV density map).

    People are splatted on a cell grid and smoothed with a Gaussian (split_sigma m). Every cell climbs to its
    highest 8-neighbour until a local maximum, which gives one basin per density peak. Two touching basins are
    merged while the density on their shared border (saddle) is >= split_valley x the lower of the two peaks,
    i.e. a part is kept separate only where the crowd density drops by more than (1 - split_valley) between them.
    Parts smaller than min_people are merged into the neighbour with the highest saddle."""
    P = bev[idx]
    wp = np.ones(len(idx)) if w is None else w[idx]
    c = cfg.split_cell
    pad = 3 * cfg.split_sigma
    o = P.min(0) - pad
    gx = ((P[:, 0] - o[0]) / c).astype(np.int64)
    gz = ((P[:, 1] - o[1]) / c).astype(np.int64)
    H, W = int(gz.max() + pad / c) + 2, int(gx.max() + pad / c) + 2
    D = np.zeros((H, W), np.float32)
    np.add.at(D, (gz, gx), wp.astype(np.float32))
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
        ids, inv = np.unique(lab, return_inverse=True)
        cnt = np.bincount(inv, weights=wp)
        small = ids[cnt < cfg.min_people - 1e-6]
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


def _size_split(idx, bev, cfg):
    """Partition a crowd wider than max_extent into compact groups of about group_area m^2 (k-means on the
    ground plane, farthest-point initialisation, deterministic). Used where a crowd is uniformly dense and has no
    density valley to split at, so that one plaza-wide crowd is reported as neighbouring groups."""
    P = bev[idx]
    if np.ptp(P, 0).max() <= cfg.max_extent:
        return [idx]
    c = 1.0
    g = np.unique(np.floor(P / c).astype(np.int64), axis=0)           # occupied 1 m cells
    k = int(min(len(idx) // max(cfg.min_people, 1), max(2, math.ceil(len(g) * c * c / cfg.group_area))))
    if k < 2:
        return [idx]
    cen = [P[np.argmin(P[:, 1])]]                                       # nearest person first
    dmin = np.full(len(P), np.inf)
    for _ in range(k - 1):
        dmin = np.minimum(dmin, ((P - cen[-1]) ** 2).sum(1))
        cen.append(P[np.argmax(dmin)])
    cen = np.array(cen)
    for _ in range(15):
        lab = ((P[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        new = np.array([P[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        if np.allclose(new, cen, atol=1e-3):
            break
        cen = new
    lab = ((P[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
    # parts below min_people join the nearest other centre
    while True:
        cnt = np.bincount(lab, minlength=k)
        small = [j for j in range(k) if 0 < cnt[j] < cfg.min_people]
        if not small:
            break
        j = small[0]
        dd = ((cen - cen[j]) ** 2).sum(1)
        dd[j] = np.inf
        dd[cnt == 0] = np.inf
        lab[lab == j] = int(np.argmin(dd))
    return [idx[lab == j] for j in np.unique(lab)]


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
            parts = _density_split(idx, bev, cfg) if big else [idx]
            if cfg.max_extent > 0:
                parts = [q for part in parts for q in _size_split(part, bev, cfg)]
            groups += parts
    groups.sort(key=lambda g: -len(g))
    for k, g in enumerate(groups):
        lab[g] = k
    return lab


def _disc_mask(pts, radius, origin, cell, shape):
    m = np.zeros(shape, np.uint8)
    c = np.round((pts - origin) / cell).astype(np.int32)
    r = np.maximum(1, np.round(np.broadcast_to(radius, len(pts)) / cell).astype(np.int32))
    cr = np.unique(np.concatenate([c, r[:, None]], 1), axis=0)
    for x, y, rr in cr:
        cv2.circle(m, (int(x), int(y)), int(rr), 1, -1)
    return m


def build_clusters(bev, sigma, uv, cam, img_wh, net_wh, cfg: ClusterCfg = ClusterCfg(), device=None):
    """Single frame: cluster labels + per-cluster occupied area, BEV outline, image polygon and risk level."""
    labels = cluster_people(bev, sigma, cfg, device)
    groups = [np.flatnonzero(labels == c) for c in range(labels.max() + 1 if len(labels) else 0)]
    out = _geometry(groups, groups, bev, uv, cam, img_wh, cfg)
    for c in out:
        c.rho = c.density
        c.level = risk_level(c.rho, cfg.risk)
    return labels, out


def _geometry(members, geo, bev_g, uv_g, cam, img_wh, cfg):
    """Clusters from member indices (current frame) and region point indices into bev_g / uv_g."""
    W, H = img_wh
    out = []
    f_img = (W * 0.5) / math.tan(float(cam[0]) * 0.5)
    drop = max(float(cam[3]) - HEAD_HEIGHT, 0.1)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(cfg.close / cfg.cell) + 1,) * 2)
    for mem, g in zip(members, geo):
        P = bev_g[g]
        pad = cfg.margin + cfg.close + cfg.cell
        o = P.min(0) - pad
        shape = (int(math.ceil((P[:, 1].max() + pad - o[1]) / cfg.cell)) + 1,
                 int(math.ceil((P[:, 0].max() + pad - o[0]) / cfg.cell)) + 1)
        m = cv2.morphologyEx(_disc_mask(P, cfg.margin, o, cfg.cell, shape), cv2.MORPH_CLOSE, k_close)
        area = float(m.sum()) * cfg.cell * cfg.cell
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bev_polys = [cv2.approxPolyDP(q, 0.75, True)[:, 0].astype(np.float64) * cfg.cell + o for q in cs if len(q) >= 3]
        out.append(Cluster(members=mem, area=area, bev_polys=bev_polys, img_polys=[], geo=g))
    _image_polygons(out, uv_g, bev_g, f_img, drop, (W, H), cfg)
    return out


def _image_polygons(clusters, uv, bev, f_img, drop, img_wh, cfg):
    """Image overlay regions that tile without overlap.

    Each cluster: discs of `margin` metres (at each person's distance) around its detected head pixels, closed
    at the same scale, on one down-scaled full-image raster. Drawn from the pixels, so a person whose metric
    position is off still gets the outline at the right place. Where regions of neighbouring clusters overlap,
    every raster cell goes to the cluster of the nearest clustered head (Voronoi split), so adjacent groups of a
    partitioned crowd share a border instead of covering each other."""
    if not clusters:
        return
    W, H = img_wh
    ds = max(cfg.img_ds, int(round(max(W, H) / 480)))
    gh, gw = H // ds + 1, W // ds + 1
    own = np.full((gh, gw), -1, np.int32)
    nown = np.zeros((gh, gw), np.uint8)
    regs = []
    for k, c in enumerate(clusters):
        U, P = uv[c.geo], bev[c.geo]
        rpx = np.clip(f_img * cfg.margin / np.sqrt((P ** 2).sum(1) + drop ** 2), 3.0, 0.25 * max(W, H))
        kc = max(1, int(round(float(np.median(rpx)) / ds)))
        pad = int(math.ceil(rpx.max() / ds)) + kc + 2               # region of interest on the raster
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
        idx = np.concatenate([c.geo for c in clusters])
        lab = np.concatenate([np.full(len(c.geo), k) for k, c in enumerate(clusters)])
        yy, xx = np.nonzero(over)
        _, nn = cKDTree(uv[idx] / ds).query(np.stack([xx, yy], 1))
        own[yy, xx] = lab[nn]
    for k, c in enumerate(clusters):
        m = ((own == k) & (regs[k] > 0)).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c.img_polys = [np.round(cv2.approxPolyDP(q, 0.6, True)[:, 0] * ds).astype(np.int32) for q in cs
                       if len(q) >= 3 and cv2.contourArea(q) >= 2]


@dataclass
class TrackCfg:
    window: int = 8           # frames pooled for clustering (static camera)
    anchor_alpha: float = 0.3  # per-frame step of a group centre towards its new position
    anchor_reach: float = 2.0  # an old centre belongs to a crowd if a crowd point is within this (m)
    keep: int = 30            # frames an unused group centre is remembered (id comes back if the group does)
    k_hyst: float = 0.5       # group-count hysteresis, in groups of group_area
    rho_alpha: float = 0.15   # density smoothing (EMA per frame)
    level_hyst: float = 0.1   # a level drops only below (1 - level_hyst) x its threshold
    stick: float = 1.0        # a person changes group only if another centre is this much closer (m)
    keep_people: float = 4.0  # an existing cluster survives down to this many people (new ones need min_people)
    relax_nb: float = 0.5     # core test relaxed by this many neighbours for people already in a cluster
    pool_cell: float = 0.25   # older frames' points are merged per BEV cell (m) before clustering
    match_px: float = 0.01    # head match to the previous frame: tolerance x image long side


class ClusterTracker:
    """Temporally stable clustering for a static camera (video).

    Per-frame clustering flickers: detections near the threshold come and go, so the core / min_people tests
    and the size split change from frame to frame. The tracker
      1. clusters the people of the last `window` frames together, each point weighted 1/window (the core and
         min_people tests become per-frame averages, the density map is smoother);
      2. keeps the group centres of the size split between frames: k-means starts from the previous centres,
         a centre moves at most anchor_alpha of the way per frame, and the number of groups changes only when
         the occupied area is k_hyst groups beyond it; the centre id is the cluster's track id;
      3. remembers each person's group (head matched to the previous frame): a person stays in its group unless
         another centre is `stick` m closer, and existing clusters keep relaxed core / size tests
         (relax_nb, keep_people), so groups do not blink at the thresholds;
      4. smooths each tracked cluster's density (EMA) and applies hysteresis to its risk level.
    Regions are drawn from the pooled points (current and recent heads), current people get the labels."""

    def __init__(self, cfg: ClusterCfg = ClusterCfg(), tcfg: TrackCfg = TrackCfg(), device=None):
        self.cfg, self.t, self.device = cfg, tcfg, device
        self.buf = deque(maxlen=tcfg.window)
        self.anchors = {}          # tid -> [pos (2,), frames unused]
        self.state = {}            # tid -> [rho, level]
        self.next_id = 0

    def _new(self, pos):
        tid, self.next_id = self.next_id, self.next_id + 1
        self.anchors[tid] = [np.asarray(pos, np.float64).copy(), 0]
        return tid

    def _owners(self, parts, B):
        """Part index of every remembered centre (nearest crowd point within anchor_reach), -1 = none."""
        if not self.anchors or not parts:
            return {}
        idx = np.concatenate(parts)
        pid = np.concatenate([np.full(len(q), k) for k, q in enumerate(parts)])
        ids = list(self.anchors)
        dd, nn = cKDTree(B[idx]).query(np.array([self.anchors[a][0] for a in ids]))
        return {a: (int(pid[j]) if d <= self.t.anchor_reach else -1) for a, d, j in zip(ids, dd, nn)}

    def _split(self, part, B, w, inside, prior):
        """Groups (point index arrays) and their track ids for one connected part (inside: its old centres)."""
        cfg, t = self.cfg, self.t
        P, wp = B[part], w[part]
        k = 1
        if cfg.max_extent > 0 and np.ptp(P, 0).max() > cfg.max_extent:
            k_real = len(np.unique(np.floor(P).astype(np.int64), axis=0)) / cfg.group_area
            lo, hi = max(2, math.ceil(k_real - t.k_hyst)), max(2, math.ceil(k_real + t.k_hyst))
            k = int(np.clip(len(inside), lo, hi)) if inside else max(2, math.ceil(k_real))
            k = max(1, min(k, int(wp.sum() // max(cfg.min_people, 1))))
        cen_w = (P * wp[:, None]).sum(0) / wp.sum()
        if k == 1:
            if inside:
                tid = min(inside, key=lambda a: ((self.anchors[a][0] - cen_w) ** 2).sum())
                a = self.anchors[tid][0]
                a += t.anchor_alpha * (cen_w - a)
            else:
                tid = self._new(cen_w)
            return [part], [tid]
        ids = list(inside)
        while len(ids) > k:                               # too many old centres: drop the lightest group
            lab = ((P[:, None] - np.array([self.anchors[a][0] for a in ids])[None]) ** 2).sum(-1).argmin(1)
            ids.pop(int(np.argmin(np.bincount(lab, weights=wp, minlength=len(ids)))))
        C = [self.anchors[a][0].copy() for a in ids]
        fresh = []
        while len(C) < k:                                 # new centres: farthest point from the existing ones
            dmin = ((P[:, None] - np.array(C)[None]) ** 2).sum(-1).min(1) if C else ((P - cen_w) ** 2).sum(1)
            C.append(P[int(np.argmax(dmin))].copy())
            fresh.append(len(C) - 1)
        C = np.array(C)
        for _ in range(5):                                # weighted k-means from the previous centres
            lab = ((P[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            sw = np.bincount(lab, weights=wp, minlength=k)
            for j in range(k):
                if sw[j] > 0:
                    C[j] = (P[lab == j] * wp[lab == j, None]).sum(0) / sw[j]
        tids = []
        for j in range(k):
            if j < len(ids):
                a = self.anchors[ids[j]][0]
                a += t.anchor_alpha * (C[j] - a)          # damped move of an old centre
                tids.append(ids[j])
            else:
                tids.append(self._new(C[j]))
        A = np.array([self.anchors[a][0] for a in tids])
        D = np.sqrt(((P[:, None] - A[None]) ** 2).sum(-1))
        pr = prior[part]
        for j, tid in enumerate(tids):                    # stickiness: own previous group is preferred
            D[pr == tid, j] -= t.stick
        lab = D.argmin(1)
        while True:                                        # groups below min_people join the nearest centre
            sw = np.bincount(lab, weights=wp, minlength=k)
            small = [j for j in range(k) if 0 < sw[j] < cfg.min_people - 1e-6]
            if not small:
                break
            j = small[0]
            dd = ((A - A[j]) ** 2).sum(1)
            dd[j] = np.inf
            dd[sw == 0] = np.inf
            lab[lab == j] = int(np.argmin(dd))
        return [part[lab == j] for j in range(k) if (lab == j).any()], [tids[j] for j in range(k) if (lab == j).any()]

    def _pool(self):
        """Current points first (one each, weight 1/K), then the older frames merged per BEV cell."""
        K = len(self.buf)
        cur = self.buf[0]
        n = len(cur["bev"])
        old = list(self.buf)[1:]
        B, S, U, T, w = [cur["bev"]], [cur["sigma"]], [cur["uv"]], [cur["prior"]], [np.full(n, 1.0 / K)]
        if old and sum(len(f["bev"]) for f in old):
            ob = np.concatenate([f["bev"] for f in old]).reshape(-1, 2)
            key = np.floor(ob / self.t.pool_cell).astype(np.int64)
            _, first, inv, cnt = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
            m = len(cnt)

            def mean(x):
                x = x.reshape(len(inv), -1)
                o = np.zeros((m, x.shape[1]))
                np.add.at(o, inv, x)
                return o / cnt[:, None]
            B.append(mean(ob))
            S.append(mean(np.concatenate([f["sigma"] for f in old]))[:, 0])
            U.append(mean(np.concatenate([f["uv"] for f in old]).astype(np.float64)))
            T.append(np.concatenate([f["tid"] for f in old])[first])     # newest occurrence (buffer is newest first)
            w.append(cnt / K)
        return (np.concatenate(B).reshape(-1, 2), np.concatenate(S), np.concatenate(U).reshape(-1, 2),
                np.concatenate(T).astype(np.int64), np.concatenate(w), n)

    def __call__(self, bev, sigma, uv, cam, img_wh):
        cfg, t = self.cfg, self.t
        bev, sigma, uv = np.asarray(bev, np.float64).reshape(-1, 2), np.asarray(sigma, np.float64), np.asarray(uv).reshape(-1, 2)
        prior = -np.ones(len(bev), np.int64)
        if self.buf and len(self.buf[0]["uv"]) and len(uv):     # previous group of each person (mutual nearest head)
            pu = self.buf[0]["uv"]
            d01, j = cKDTree(pu).query(uv)
            _, i = cKDTree(uv).query(pu)
            ok = (d01 < t.match_px * max(img_wh)) & (i[j] == np.arange(len(uv)))
            prior[ok] = self.buf[0]["tid"][j[ok]]
        prior[(prior >= 0) & ~np.isin(prior, list(self.anchors))] = -1
        self.buf.appendleft({"bev": bev, "sigma": sigma, "uv": uv, "prior": prior, "tid": -np.ones(len(bev), np.int64)})
        B, S, U, T, w, n = self._pool()
        live = np.array([tid for tid, (_, miss) in self.anchors.items() if miss == 0], np.int64)
        T[~np.isin(T, live)] = -1
        labels = -np.ones(n, np.int64)
        groups, tids, used = [], [], set()
        if len(B) >= 2:
            d = np.hypot(B[:, 0], B[:, 1])
            s = np.clip(cfg.rad_k * d * S, 1.0, cfg.rad_max)
            raw = _dbscan(B, d, s, cfg.eps, cfg.min_nb, self.device, w, np.where(T >= 0, t.relax_nb, 0.0))
            parts = []
            for c in np.unique(raw[raw >= 0]):
                idx = np.flatnonzero(raw == c)
                need = t.keep_people if (T[idx] >= 0).any() else cfg.min_people
                if w[idx].sum() >= need - 1e-6:
                    big = cfg.split_valley < 1.0 and np.ptp(B[idx], 0).max() > cfg.split_extent
                    parts += _density_split(idx, B, cfg, w) if big else [idx]
            parts.sort(key=lambda q: -w[q].sum())
            own = self._owners(parts, B)
            for k, part in enumerate(parts):
                g, ti = self._split(part, B, w, [a for a, o in own.items() if o == k], T)
                for q, tid in zip(g, ti):
                    if (q < n).any():
                        groups.append(q)
                        tids.append(tid)
                        used.add(tid)
        for tid in list(self.anchors):
            if tid in used:
                self.anchors[tid][1] = 0
            else:
                self.anchors[tid][1] += 1
                if self.anchors[tid][1] > t.keep:
                    del self.anchors[tid]
                    self.state.pop(tid, None)
        order = sorted(range(len(groups)), key=lambda j: -int((groups[j] < n).sum()))
        groups, tids = [groups[j] for j in order], [tids[j] for j in order]
        members = [g[g < n] for g in groups]
        for k, m in enumerate(members):
            labels[m] = k
            self.buf[0]["tid"][m] = tids[k]
        out = _geometry(members, groups, B, U, cam, img_wh, cfg)
        for c, tid, g in zip(out, tids, groups):
            c.tid = tid
            rho_now = float(w[g].sum()) / max(c.area, 1e-6)            # frame-averaged count / pooled area
            st = self.state.get(tid)
            if st is None:
                st = self.state[tid] = [rho_now, risk_level(rho_now, cfg.risk)]
            else:
                st[0] += t.rho_alpha * (rho_now - st[0])
                st[1] = risk_level(st[0], cfg.risk, st[1], t.level_hyst)
            c.rho, c.level = st
        return labels, out
