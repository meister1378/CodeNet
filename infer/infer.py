"""CoDE-Net inference: image / folder / video -> people (pixel + metric BEV), crowd clusters, renders.

    python infer.py input.jpg   --out out/            # overlay + BEV image, json
    python infer.py frames_dir/ --out out/
    python infer.py video.mp4   --out out/ --fixed-camera   # overlay | BEV video, camera median over frames

Python:
    from infer import CodeNetPredictor
    pred = CodeNetPredictor("../checkpoints/codenet_m.pth.enc")
    res = pred(bgr)                     # dict: uv, score, bev, sigma, camera, labels, clusters
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "model"))
sys.path.insert(0, _HERE)
from ckpt_crypto import load_checkpoint  # noqa: E402
from model import CodeNet  # noqa: E402
from postprocess import CameraSmoother, ClusterCfg, ClusterTracker, TrackCfg, build_clusters  # noqa: E402
from visualize import BevCanvas, cluster_table, compose, draw_overlay  # noqa: E402
import visualize as vis  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CodeNetPredictor:
    """Loads CoDE-Net once; __call__(bgr) runs the network (people + camera + BEV lift) and the clustering.

    long_side: network input long side (rounded to /32). thr: detection score threshold.
    fixed_camera: median of the camera estimate over the last `camera_window` frames is used for the lift
    (static camera, video). camera: optional fixed (fov, pitch, roll, height) [rad, rad, rad, m] override.
    track: temporally stable clustering for consecutive frames of a static camera (ClusterTracker); call
    reset() between unrelated videos.
    """

    def __init__(self, ckpt_path, device="cuda", preset="m", thr=0.3, long_side=1280, fp16=True,
                 fixed_camera=False, camera_window=25, camera=None, cluster_cfg: ClusterCfg | None = None,
                 track=False, track_cfg: TrackCfg | None = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        net = CodeNet(preset)
        sd = load_checkpoint(ckpt_path, map_location="cpu")
        missing, unexpected = net.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("aux_") for k in missing), (missing, unexpected)
        self.net = net.to_deploy().to(self.device).eval()
        self.thr, self.long_side = thr, long_side
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        self.cluster_cfg = cluster_cfg or ClusterCfg()
        self.track_cfg = track_cfg or TrackCfg()
        self.tracker = ClusterTracker(self.cluster_cfg, self.track_cfg, self.device) if track else None
        self.smoother = CameraSmoother(camera_window) if fixed_camera else None
        self.camera_fixed = None if camera is None else np.asarray(camera, np.float64)
        self._cam_raw = None
        orig = self.net.camera

        def camera_hook(c5):                      # replaces the per-frame camera used by the in-graph lift
            cam = orig(c5)
            v = torch.stack([cam["fov"], cam["pitch"], cam["roll"], cam["height"]], 1)[0].float().cpu().numpy()
            self._cam_raw = v
            use = self.camera_fixed if self.camera_fixed is not None else self.smoother(v) if self.smoother else None
            if use is None:
                return cam
            t = torch.as_tensor(use, dtype=cam["fov"].dtype, device=cam["fov"].device)[None]
            return {"fov": t[:, 0], "pitch": t[:, 1], "roll": t[:, 2], "height": t[:, 3]}

        self.net.camera = camera_hook

    def preprocess(self, bgr):
        """uint8 BGR -> normalised RGB tensor at the network size (area down-sampling on the device)."""
        h, w = bgr.shape[:2]
        s = self.long_side / max(h, w)
        nh, nw = int(round(h * s / 32)) * 32, int(round(w * s / 32)) * 32
        x = torch.from_numpy(np.ascontiguousarray(bgr)).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1)[None].flip(1).float().div_(255.0)            # RGB, 0..1
        mode = "area" if nh < h else "bilinear"
        x = F.interpolate(x, size=(nh, nw), mode=mode, **({} if mode == "area" else {"align_corners": False}))
        x = (x - self._mean) / self._std
        return x, (nw, nh)

    @torch.no_grad()
    def detect(self, bgr):
        H, W = bgr.shape[:2]
        x, (nw, nh) = self.preprocess(bgr)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            top, uv, size, bev, sigma, valid, cam = self.net(x)
        m = (top[0] > self.thr) & (valid[0] > 0.5)
        uv = uv[0][m].float().cpu().numpy() * [W / nw, H / nh]
        return {"uv": uv, "score": top[0][m].float().cpu().numpy(),
                "head_px": size[0][m].float().cpu().numpy() * (W / nw),
                "bev": bev[0][m].float().cpu().numpy().astype(np.float64),
                "sigma": sigma[0][m].float().cpu().numpy().astype(np.float64),
                "camera": cam[0].float().cpu().numpy().astype(np.float64), "camera_raw": self._cam_raw,
                "img_wh": (W, H), "net_wh": (nw, nh)}

    def reset(self):
        if self.tracker is not None:
            self.tracker = ClusterTracker(self.cluster_cfg, self.track_cfg, self.device)
        if self.smoother is not None:
            self.smoother = CameraSmoother(self.smoother.buf.maxlen)

    def cluster(self, r):
        """Adds labels / clusters (tracked when track=True) and the risk thresholds to a detect() result."""
        if self.tracker is not None:
            r["labels"], r["clusters"] = self.tracker(r["bev"], r["sigma"], r["uv"], r["camera"], r["img_wh"])
        else:
            r["labels"], r["clusters"] = build_clusters(r["bev"], r["sigma"], r["uv"], r["camera"], r["img_wh"],
                                                        r["net_wh"], self.cluster_cfg, self.device)
        r["risk"] = tuple(self.cluster_cfg.risk)
        return r

    def __call__(self, bgr):
        return self.cluster(self.detect(bgr))


def to_json(r):
    cam = r["camera"]
    return {"camera": {"fov_deg": math.degrees(cam[0]), "pitch_deg": math.degrees(cam[1]),
                       "roll_deg": math.degrees(cam[2]), "height_m": float(cam[3])},
            "people": [{"u": float(u), "v": float(v), "score": float(s), "x_m": float(b[0]), "z_m": float(b[1]),
                        "log_sigma": float(sg), "cluster": int(l)}
                       for (u, v), s, b, sg, l in zip(r["uv"], r["score"], r["bev"], r["sigma"], r["labels"])],
            "clusters": [{"id": k + 1, "count": c.count, "area_m2": c.area, "density_per_m2": c.rho, "level": vis.LEVEL_NAME[c.level], "track_id": c.tid,
                          "polygon_m": [p.round(2).tolist() for p in c.bev_polys],
                          "polygon_px": [p.tolist() for p in c.img_polys]} for k, c in enumerate(r["clusters"])]}


def render(r, img, canvas: BevCanvas, title=""):
    ov = draw_overlay(img, r["uv"], r["labels"], r["clusters"])
    bv = canvas.fit(r["bev"]).draw(r["bev"], r["labels"], r["clusters"], r["camera"])
    return compose(ov, bv, cluster_table(r["clusters"], len(r["uv"])), title or "top view (BEV)", r["camera"],
                   r.get("risk", (2.0, 4.0)))


def run_images(pred, paths, out, size):
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print("skip", p)
            continue
        t = time.time()
        r = pred(img)
        dt = time.time() - t
        cv2.imwrite(str(out / f"{p.stem}_codenet.jpg"), render(r, img, BevCanvas(size)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        json.dump(to_json(r), open(out / f"{p.stem}_codenet.json", "w"))
        print(f"{p.name}: {len(r['uv'])} people, {len(r['clusters'])} clusters, {dt * 1000:.0f} ms")


def run_video(pred, path, out, size, start=0.0, seconds=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    n_max = int(round(seconds * fps)) if seconds else None
    canvas, writer, n, t_net, t_all = BevCanvas(size), None, 0, 0.0, 0.0
    while n_max is None or n < n_max:
        ok, img = cap.read()
        if not ok:
            break
        t0 = time.time()
        r = pred.detect(img)
        if pred.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        pred.cluster(r)
        t2 = time.time()
        if n >= 3:                                     # skip warm-up frames in the timing
            t_net, t_all = t_net + (t1 - t0), t_all + (t2 - t0)
        if canvas.z_max is None and n == 0:
            canvas.fit(r["bev"])
        frame = render(r, img, canvas)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out / f"{Path(path).stem}_codenet.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    k = max(1, n - 3)
    print(f"{Path(path).name}: {n} frames, network {1000 * t_net / k:.1f} ms/frame, "
          f"network + clustering {1000 * t_all / k:.1f} ms/frame")


def main():
    ap = argparse.ArgumentParser(description="CoDE-Net inference")
    ap.add_argument("input", help="image, folder of images, or video")
    ap.add_argument("--out", default="out")
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "..", "checkpoints", "codenet_m.pth.enc"))
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--fixed-camera", action="store_true", help="median camera over frames (static camera)")
    ap.add_argument("--camera-window", type=int, default=25, help="frames in the camera median (--fixed-camera)")
    ap.add_argument("--camera", type=float, nargs=4, metavar=("FOV_DEG", "PITCH_DEG", "ROLL_DEG", "HEIGHT_M"),
                    help="use a known camera instead of the estimate")
    ap.add_argument("--start", type=float, default=0.0, help="video start (s)")
    ap.add_argument("--seconds", type=float, default=None, help="video duration (s)")
    ap.add_argument("--bev-size", type=int, default=720)
    ap.add_argument("--risk", type=float, nargs=2, default=[2.0, 4.0], metavar=("CAUTION", "WARNING"),
                    help="cluster density thresholds (people/m2) for caution / warning colours")
    ap.add_argument("--no-track", action="store_true",
                    help="cluster every video frame independently (default with --fixed-camera / --camera: tracked)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    cam = None
    if a.camera:
        cam = [math.radians(a.camera[0]), math.radians(a.camera[1]), math.radians(a.camera[2]), a.camera[3]]
    src = Path(a.input)
    video = src.is_file() and src.suffix.lower() not in IMG_EXT
    track = video and (a.fixed_camera or cam is not None) and not a.no_track
    pred = CodeNetPredictor(a.ckpt, device="cpu" if a.cpu else "cuda", thr=a.thr, long_side=a.long_side,
                            fixed_camera=a.fixed_camera, camera_window=a.camera_window, camera=cam,
                            cluster_cfg=ClusterCfg(risk=tuple(a.risk)), track=track)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if video:
        run_video(pred, src, out, a.bev_size, a.start, a.seconds)
    else:
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
        run_images(pred, paths, out, a.bev_size)


if __name__ == "__main__":
    main()
