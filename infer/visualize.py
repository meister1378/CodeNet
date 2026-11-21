"""Rendering (OpenCV only): image overlay with cluster polygons and a metric top view (BEV) panel."""
from __future__ import annotations

import math

import cv2
import numpy as np

# 10 well-separated colours (BGR); clusters cycle through them, people outside clusters are grey
PALETTE = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
           (75, 86, 140), (194, 119, 227), (34, 189, 188), (207, 190, 23), (127, 127, 127)]
GREY = (170, 170, 170)


def colour(i: int):
    return GREY if i < 0 else PALETTE[i % 9]


def _text(img, s, org, scale, colour_, thick=1, bg=None):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour_, thick, cv2.LINE_AA)


def draw_overlay(img, uv, labels, clusters, alpha=0.35):
    """Cluster polygons (filled, alpha) + head dots coloured by cluster + C<k> tags with count and density."""
    out = img.copy()
    H, W = img.shape[:2]
    lw = max(1, round(max(H, W) / 900))
    fill = img.copy()
    for k, c in enumerate(clusters):
        cv2.fillPoly(fill, c.img_polys, colour(k))
    cv2.addWeighted(fill, alpha, out, 1 - alpha, 0, out)
    for k, c in enumerate(clusters):
        cv2.polylines(out, c.img_polys, True, colour(k), lw + 1, cv2.LINE_AA)
    r = max(2, round(max(H, W) / 600))
    for (x, y), l in zip(uv.astype(int), labels):
        cv2.circle(out, (int(x), int(y)), r, colour(int(l)), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)
    fs = max(0.45, max(H, W) / 2600)
    for k, c in enumerate(clusters):
        top = c.members[np.argmin(uv[c.members, 1])]
        _text(out, f"C{k + 1} {c.count}", (uv[top, 0] - 10, uv[top, 1] - 3 * r), fs, (255, 255, 255), max(1, lw), colour(k))
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

    def draw(self, bev, labels, clusters, cam, title=""):
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
        for k, c in enumerate(clusters):
            for q in c.bev_polys:
                p = np.round(self.to_px(q)).astype(np.int32)
                ov = img.copy()
                cv2.fillPoly(ov, [p], colour(k))
                cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
                cv2.polylines(img, [p], True, colour(k), 1, cv2.LINE_AA)
        P = np.round(self.to_px(bev)).astype(int) if len(bev) else np.zeros((0, 2), int)
        for (x, y), l in zip(P, labels):
            if 0 <= x < S and 0 <= y < S:
                cv2.circle(img, (int(x), int(y)), 2, colour(int(l)), -1, cv2.LINE_AA)
        for k, c in enumerate(clusters[:20]):
            q = self.to_px(bev[c.members].mean(0))
            if 0 <= q[0] < S and 0 <= q[1] < S:
                _text(img, f"C{k + 1}", (q[0] + 6, q[1]), 0.4, (0, 0, 0))
        cv2.drawMarker(img, o, (0, 0, 220), cv2.MARKER_TRIANGLE_UP, 14, 2)
        return img


def cluster_table(clusters, n_people, max_rows=6):
    """Fixed number of lines (constant frame size for video)."""
    rows = [f"{n_people} people, {len(clusters)} clusters"]
    rows += [f"C{k + 1}: {c.count:4d} people  {c.area:6.0f} m2  {c.density:4.2f}/m2" for k, c in enumerate(clusters[:max_rows])]
    return rows + [""] * (max_rows + 1 - len(rows))


def compose(overlay, bev_img, rows, title="top view (BEV)", cam=None):
    """Overlay (left) | header band (title, camera, cluster table) above the BEV panel (right)."""
    S = bev_img.shape[1]
    lines = [title] + ([f"camera: fov {math.degrees(cam[0]):.0f} deg  tilt {math.degrees(cam[1]):.0f} deg  "
                        f"height {cam[3]:.1f} m"] if cam is not None else []) + rows
    band = np.full((14 + 17 * len(lines), S, 3), 255, np.uint8)
    for i, r in enumerate(lines):
        c = (0, 0, 0) if i == 0 else (60, 60, 60) if (cam is not None and i == 1) else (0, 0, 160) if r is rows[0] else (0, 0, 0)
        _text(band, r, (10, 18 + 17 * i), 0.5 if i == 0 else 0.42, c, 1)
    right = np.concatenate([band, bev_img], 0)
    H = right.shape[0]
    h, w = overlay.shape[:2]
    left = cv2.resize(overlay, (int(round(w * H / h)) // 2 * 2, H), interpolation=cv2.INTER_AREA)
    return np.concatenate([left, right], 1)
