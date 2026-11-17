"""Fuse the ATD ball with the detector's ball: ATD where it has one, else the detector above CONF.

    .venv/bin/python src/ball_fuse.py [--det eda/out/ball/detections.npz]   # -> eda/atd_tactical_fused.npz
    from src.ball_fuse import load_fused   # same dict as eda.atd.load("tactical") with the fused ball
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from eda.atd import load
from src.possession import in_play
CONF = 0.5; SUP = 10; W = 5; OUT = os.path.join(ROOT, "eda/atd_tactical_fused.npz")

def support(det, F, w=W):
    """How many of the +-w neighbouring frames hold a detection within 1.5 m/frame of this one (max 2w).
    A real ball moves smoothly; a false positive jumps. On ATD-ball frames: support 10/10 -> 12% of
    detections >5 m off, support <=5 -> 66-95% off (2026-10-27)."""
    f = det[:, 0].astype(int); pos = np.full((F, 2), np.nan); pos[f] = det[:, 1:3]; sup = np.zeros(len(det), int)
    for dt in range(1, w + 1):
        for s in (-dt, dt):
            g = f + s; ok = (g >= 0) & (g < F); d = np.full(len(det), np.inf); d[ok] = np.linalg.norm(det[ok, 1:3] - pos[g[ok]], axis=1); sup += d <= 1.5 * dt + 1
    return sup

def fuse(det_path):
    d = load("tactical"); det = np.load(det_path)["det"]; ball = d["ball"].copy(); src = np.where(np.isfinite(ball[:, 0]), 1, 0).astype(np.int8)
    ok = (det[:, 3] >= CONF) & (support(det, len(ball)) >= SUP); f = det[ok, 0].astype(int); missing = ~np.isfinite(ball[f, 0])
    ball[f[missing], :2] = det[ok][missing, 1:3]; src[f[missing]] = 2
    alive = in_play(len(ball))
    print("graded-frame ball visibility: ATD %.3f -> fused %.3f (detector filled %d frames)" % ((src[alive] == 1).mean(), (src[alive] > 0).mean(), (src[alive] == 2).sum()))
    return {"players": d["players"], "ball": ball, "team": d["team"], "ball_src": src}

def load_fused():
    z = np.load(OUT); return {k: z[k] for k in z.files}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--det", default=os.path.join(ROOT, "eda/out/ball/detections.npz")); a = ap.parse_args()
    out = fuse(a.det); np.savez_compressed(OUT, **out); print("->", OUT)
