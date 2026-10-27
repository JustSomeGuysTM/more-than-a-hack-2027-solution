"""Fuse the ATD ball with the detector's ball: ATD where it has one, else the detector above CONF.

    .venv/bin/python src/ball_fuse.py [--det eda/out/ball/detections.npz]   # -> eda/atd_tactical_fused.npz
    from src.ball_fuse import load_fused   # same dict as eda.atd.load("tactical") with the fused ball
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from eda.atd import load
CONF = 0.25; OUT = os.path.join(ROOT, "eda/atd_tactical_fused.npz")

def fuse(det_path):
    d = load("tactical"); det = np.load(det_path)["det"]; ball = d["ball"].copy(); src = np.where(np.isfinite(ball[:, 0]), 1, 0).astype(np.int8)
    ok = det[:, 3] >= CONF; f = det[ok, 0].astype(int); missing = ~np.isfinite(ball[f, 0])
    ball[f[missing], :2] = det[ok][missing, 1:3]; src[f[missing]] = 2
    alive = np.zeros(len(ball), bool); alive[11185:90652] = True; alive[113742:190198] = True
    print("graded-frame ball visibility: ATD %.3f -> fused %.3f (detector filled %d frames)" % ((src[alive] == 1).mean(), (src[alive] > 0).mean(), (src[alive] == 2).sum()))
    return {"players": d["players"], "ball": ball, "team": d["team"], "ball_src": src}

def load_fused():
    z = np.load(OUT); return {k: z[k] for k in z.files}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--det", default=os.path.join(ROOT, "eda/out/ball/detections.npz")); a = ap.parse_args()
    out = fuse(a.det); np.savez_compressed(OUT, **out); print("->", OUT)
