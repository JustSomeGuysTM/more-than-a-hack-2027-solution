"""Temporal layer for the ball detector (plans/soccernet_ball_detector.md step 4): Viterbi over each frame's top-k
candidates plus a "no ball" state. A real ball moves smoothly; a false positive (head, line, spare ball) appears
alone or jumps, so it has to pay to switch the track on and off.

    .venv/bin/python src/ball_detect.py --infer --topk 5 --window 100 [--weights <pt>]   # dense candidates
    .venv/bin/python src/ball_track.py eda/out/ball/detections_top5_win100.npz          # track + score vs the ATD
    .venv/bin/python src/ball_track.py --selftest
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
P = dict(sigma=1.0, move=0.6, vmax=1.6, jump=8.0, on=4.0, off=4.0, none=1.5, min_len=5)   # detector noise m, typical ball move m/frame, max m/frame (40 m/s); costs in -log units

def track(det, p=P):
    """det rows (frame, x, y, conf, ...), any number per frame. Returns {frame: (x, y, conf)} for frames on the best path
    whose run of consecutive tracked frames is >= min_len. Frames are chained only when consecutive."""
    det = det[np.lexsort((-det[:, 3], det[:, 0]))]; fs = np.unique(det[:, 0].astype(int)); out = {}
    for run in np.split(fs, np.where(np.diff(fs) != 1)[0] + 1):
        C = [det[det[:, 0] == f] for f in run]                                   # candidates per frame; state len(c) = none
        cost = np.r_[-np.log(C[0][:, 3]) + p["on"], p["none"]]; back = []
        for t in range(1, len(run)):
            a, b = C[t - 1], C[t]; na, nb = len(a), len(b)
            d = np.linalg.norm(a[:, None, 1:3] - b[None, :, 1:3], axis=2)
            move = np.where(d <= p["vmax"] + 2 * p["sigma"], 0.5 * (d / (p["sigma"] + p["move"])) ** 2, p["jump"])
            T = np.zeros((na + 1, nb + 1)); T[:na, :nb] = move; T[:na, nb] = p["off"]; T[na, :nb] = p["on"]; T[na, nb] = 0
            tot = cost[:, None] + T; back.append(tot.argmin(0)); cost = tot.min(0) + np.r_[-np.log(b[:, 3]), p["none"]]
        s = int(cost.argmin()); path = [s]
        for bk in back[::-1]: s = int(bk[s]); path.append(s)
        path = path[::-1]; on = np.array([s < len(c) for s, c in zip(path, C)])
        edges = np.diff(np.r_[0, on.astype(int), 0])
        for a0, a1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if a1 - a0 >= p["min_len"]:
                for t in range(a0, a1): out[int(run[t])] = tuple(C[t][path[t], 1:4])
    return out

GRADED = ((11185, 90652), (113742, 190198))   # graded play (src/ball_fuse.py); outside it, warm-up balls make the ATD a poor reference

def score(out, step=50, offset=25, window=100, graded=True):
    """Same frames and metric as `src/ball_detect.py --check --step 50 --offset 25 --window 100 [--graded]`."""
    from eda.atd import load; from src.ball_detect import homographies
    atd = load("tactical")["ball"]; H = homographies()
    fr = np.array([f for f in range(offset, len(atd), step) if np.isfinite(H[f]).all() and f % 500 < window and (not graded or any(a <= f < b for a, b in GRADED))]); seen = np.isfinite(atd[fr, 0])
    hit = np.array([f in out for f in fr]); err = np.array([np.hypot(out[f][0] - atd[f, 0], out[f][1] - atd[f, 1]) for f in fr[seen & hit]])
    return f"{len(fr)} frames: detects on ATD-ball frames {hit[seen].mean():.0%}, median error {np.median(err):.1f} m, >5 m {np.mean(err > 5):.0%} | detects on ATD-missing frames {hit[~seen].mean():.0%}"

def selftest():
    r = np.random.default_rng(0); rows = []
    for f in range(200):
        x = 30 + 0.5 * f; rows.append((f, x + r.normal(0, 0.3), 20 + r.normal(0, 0.3), r.uniform(0.3, 0.6)))   # true ball, modest conf
        if f % 7 == 0: rows.append((f, r.uniform(0, 105), r.uniform(0, 68), 0.9))                         # confident one-frame false positives
    for f in range(300, 302): rows.append((f, 50.0, 30.0, 0.95))                                          # isolated 2-frame blip
    out = track(np.array(rows))
    err = [abs(out[f][0] - (30 + 0.5 * f)) for f in range(200) if f in out]
    assert len(err) >= 190 and max(err) < 2, (len(err), max(err)); assert 300 not in out
    print("selftest ok:", len(err), "of 200 frames on the true track; false positives and the blip dropped")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("det", nargs="?"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    else:
        det = np.load(a.det)["det"]; top1 = {int(r[0]): tuple(r[1:4]) for r in det[::-1]}
        for g in (True, False):
            print("graded play only" if g else "all frames (incl. warm-up)")
            for c in (0.25, 0.4, 0.6): print(f"  top-1, conf >= {c}:", score({f: v for f, v in top1.items() if v[2] >= c}, graded=g))
            print("  tracked          :", score(track(det), graded=g))
