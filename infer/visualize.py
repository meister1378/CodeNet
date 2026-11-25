"""Rendering (OpenCV only): image overlay with crowd zones and a metric top view (BEV) panel."""
from __future__ import annotations

import math

import cv2
import numpy as np

# Zones are green; caution / warning areas inside them yellow / red. Heads in zones white, others grey (BGR)
GREY = (170, 170, 170)
LEVEL_NAME = ("safe", "caution", "warning")
LEVEL_COLOUR = ((60, 175, 40), (0, 200, 245), (40, 40, 220))     # green, yellow, red


def _edge(col):
    """Darker shade of the level colour for outlines, so neighbouring regions of one colour stay separable."""
    return tuple(int(v * 0.55) for v in col)


def _text(img, s, org, scale, colour_, thick=1, bg=None):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour_, thick, cv2.LINE_AA)


def _layers(zones):
    """(colour, polygons) from the bottom up: whole zones, then caution, then warning patches."""
    out = [(LEVEL_COLOUR[0], z, "z") for z in zones]
    for L in (1, 2):
        out += [(LEVEL_COLOUR[L], p, "p") for z in zones for p in z.hot if p.level == L]
    return out


def draw_overlay(img, uv, labels, zones, alpha=0.35):
    """Zone regions (filled, alpha) with caution / warning patches + head dots."""
    out = img.copy()
    H, W = img.shape[:2]
    lw = max(1, round(max(H, W) / 900))
    fill = img.copy()
    layers = _layers(zones)
    for col, r, _ in layers:
        cv2.fillPoly(fill, r.img_polys, col)
    cv2.addWeighted(fill, alpha, out, 1 - alpha, 0, out)
    for col, r, kind in layers:
        cv2.polylines(out, r.img_polys, True, _edge(col), lw + (kind == "z"), cv2.LINE_AA)
    r = max(2, round(max(H, W) / 600))
    for (x, y), l in zip(uv.astype(int), labels):
        cv2.circle(out, (int(x), int(y)), r, GREY if l < 0 else (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)
    return out


class BevCanvas:
    """Camera-centred top view: X right, Z forward (up on the canvas), metres, 10 m range rings."""

    def __init__(self, size=720, z_max=None, margin=36):
        self.size, self.z_max, self.m = size, z_max, margin

    def fit(self, bev):
        """Range from the 98th percentile of Z (rounded up to 10 m); fixed afterwards for video."""
        if self.z_max is None:
            z = np.percentile(bev[:, 1], 98) * 1.1 if len(bev) else 30.0
            self.z_max = float(np.clip(math.ceil(z / 10.0) * 10.0, 20.0, 400.0))
        return self

    def to_px(self, P):
        s = (self.size - 2 * self.m) / self.z_max
        return np.stack([self.size / 2 + P[..., 0] * s, self.size - self.m - P[..., 1] * s], -1)

    def draw(self, bev, labels, zones, cam):
        S, zm = self.size, self.z_max
        img = np.full((S, S, 3), 255, np.uint8)
        s = (S - 2 * self.m) / zm
        o = (int(S / 2), int(S - self.m))
        step = 10 if zm <= 80 else 20 if zm <= 200 else 50
        for r in range(step, int(zm * 1.45) + 1, step):
            cv2.ellipse(img, o, (int(r * s), int(r * s)), 0, 180, 360, (230, 210, 190), 1, cv2.LINE_AA)
            if r <= zm:
                _text(img, f"{r} m", (o[0] + 4, o[1] - r * s - 3), 0.4, (180, 130, 90))
        half = float(cam[0]) / 2
        for sg in (-1, 1):
            e = self.to_px(np.array([sg * zm * 1.5 * math.tan(half), zm * 1.5]))
            cv2.line(img, o, (int(e[0]), int(e[1])), (190, 190, 190), 1, cv2.LINE_AA)
        P = np.round(self.to_px(bev)).astype(int) if len(bev) else np.zeros((0, 2), int)
        for (x, y), l in zip(P, labels):                    # people first, zones drawn over them
            if 0 <= x < S and 0 <= y < S:
                cv2.circle(img, (int(x), int(y)), 1, (150, 150, 150) if l < 0 else (70, 70, 70), -1, cv2.LINE_AA)
        for col, r, _ in _layers(zones):
            ps = [np.round(self.to_px(q)).astype(np.int32) for q in r.bev_polys]
            ov = img.copy()
            cv2.fillPoly(ov, ps, col)
            cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
            cv2.polylines(img, ps, True, _edge(col), 1, cv2.LINE_AA)
        cv2.drawMarker(img, o, (0, 0, 220), cv2.MARKER_TRIANGLE_UP, 14, 2)
        return img


def zone_table(zones, n_people, max_rows=6):
    """Header line + the most critical zones first, as (text, swatch colour); fixed number of lines for video."""
    n_lv = [sum(z.level == v for z in zones) for v in range(3)]
    inside = sum(z.count for z in zones)
    rows = [(f"{n_people} people ({inside} in {len(zones)} zones)   "
             f"warning {n_lv[2]}  caution {n_lv[1]}  safe {n_lv[0]}", None)]
    order = sorted(zones, key=lambda z: (-z.level, -z.peak))[:max_rows]
    rows += [(f"{z.count:4d} people  {z.area:6.0f} m2  mean {z.rho:4.2f}/m2  peak {z.peak:4.2f}/m2  "
              f"{LEVEL_NAME[z.level]}", LEVEL_COLOUR[z.level]) for z in order]
    return rows + [("", None)] * (max_rows + 1 - len(rows))


def compose(overlay, bev_img, rows, title="top view (BEV)", cam=None, risk=(2.0, 4.0), crowd=1.0):
    """Overlay (left) | header band (title, camera, density legend, zone table) above the BEV panel (right)."""
    S = bev_img.shape[1]
    head = [title] + ([f"camera: fov {math.degrees(cam[0]):.0f} deg  tilt {math.degrees(cam[1]):.0f} deg  "
                       f"height {cam[3]:.1f} m"] if cam is not None else [])
    n = len(head) + 1 + len(rows)
    band = np.full((14 + 17 * n, S, 3), 255, np.uint8)
    for i, r in enumerate(head):
        _text(band, r, (10, 18 + 17 * i), 0.5 if i == 0 else 0.42, (0, 0, 0) if i == 0 else (60, 60, 60), 1)
    y = 18 + 17 * len(head)                                  # legend: swatch + local density per colour
    x = 10
    for v, s_ in enumerate((f"crowd {crowd:g}-{risk[0]:g}/m2", f"caution {risk[0]:g}-{risk[1]:g}/m2", f"warning >={risk[1]:g}/m2")):
        cv2.rectangle(band, (x, y - 10), (x + 12, y + 1), LEVEL_COLOUR[v], -1)
        _text(band, s_, (x + 17, y), 0.42, (40, 40, 40), 1)
        x += 30 + cv2.getTextSize(s_, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    for i, (r, sw) in enumerate(rows):
        y = 18 + 17 * (len(head) + 1 + i)
        if sw is not None:
            cv2.rectangle(band, (10, y - 10), (22, y + 1), sw, -1)
        _text(band, r, (10 if i == 0 else 28, y), 0.42, (0, 0, 160) if i == 0 else (0, 0, 0), 1)
    right = np.concatenate([band, bev_img], 0)
    H = right.shape[0]
    h, w = overlay.shape[:2]
    left = cv2.resize(overlay, (int(round(w * H / h)) // 2 * 2, H), interpolation=cv2.INTER_AREA)
    return np.concatenate([left, right], 1)
