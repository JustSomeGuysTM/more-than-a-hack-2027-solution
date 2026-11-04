"""Learned temporal layer (plans/soccernet_ball_detector.md step 4, "learned version"): a small BiGRU reads each frame's
top-5 detector candidates and picks the ball or "none". Trained on SoccerNet clips (human truth), applied to our feed in
pixel space: SoccerNet frames are shrunk so the ball has our native size (OUR_BALL px), both at 25 fps.

    .venv/bin/python src/ball_scorer.py --cands     # current weights on every SoccerNet frame (CPU) -> eda/out/ball/sn_cands.npz
    .venv/bin/python src/ball_scorer.py --fit       # train on 47 clips, report on the last 10 (soccernet.VAL_CLIPS)
    .venv/bin/python src/ball_scorer.py --ours eda/out/ball/detections_top5_win100.npz   # score vs the ATD, as ball_track.py
    .venv/bin/python src/ball_scorer.py --selftest
"""
import argparse, glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
K = 5; DTS = (-2, -1, 1, 2); CLIP = 40.0; OUR_BALL = 4.2   # slots; neighbour frames; px distance cap; our median ball px
CANDS = os.path.join(ROOT, "eda/out/ball/sn_cands.npz"); MODEL = os.path.join(ROOT, "src/ball_scorer.pt")

def cands(weights=os.path.join(ROOT, "src/ball_yolov8n.pt")):
    """Rows (clip, frame, u, v, conf) in shrunk pixels, top-5 merged within 10 px as ball_detect.infer; truth rows
    (clip, frame, u, v, ball px). BGR, as the weights were trained (and as ball_detect.infer feeds them since 2026-11-04)."""
    import cv2; from ultralytics import YOLO; from src.soccernet import SN, ball_boxes
    m = YOLO(weights); rows, gt = [], []
    for ci, c in enumerate(sorted(glob.glob(os.path.join(SN, "SNMOT-*")))):
        b = {int(f): (x + w / 2, y + h / 2, np.sqrt(w * h)) for f, x, y, w, h in ball_boxes(c)}
        s = OUR_BALL / np.median([v[2] for v in b.values()])                           # one scale per clip: size still varies
        for f in range(1, 751):
            p = os.path.join(c, "img1", f"{f:06d}.jpg")
            if not os.path.exists(p): break
            img = cv2.resize(cv2.imread(p), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            r = m.predict(img, imgsz=640, conf=0.1, device="cpu", verbose=False)[0]; kept = []
            for (u, v), cf in sorted(zip(r.boxes.xywh.numpy()[:, :2].tolist(), r.boxes.conf.numpy().tolist()), key=lambda t: -t[1]):
                if len(kept) < K and all(np.hypot(u - a, v - bb) > 10 for a, bb, _ in kept): kept.append((u, v, cf))
            rows += [(ci, f, u, v, cf) for u, v, cf in kept]
            if f in b: gt.append((ci, f, b[f][0] * s, b[f][1] * s, b[f][2] * s))
        print(os.path.basename(c), len(rows), flush=True)
    np.savez_compressed(CANDS, det=np.array(rows, np.float32), gt=np.array(gt, np.float32))

def features(fr, uvc):
    """fr: frame per row, uvc: (u, v, conf) per row, one sequence. -> (frames, (n_frames, K*(2+2*len(DTS)))), (n_frames, K, 3).
    Per slot: present, conf, and for each neighbour frame the distance to its nearest candidate (capped, /CLIP) and that one's conf."""
    fs = np.arange(fr.min(), fr.max() + 1); S = np.full((len(fs), K, 3), np.nan)
    for i, f in enumerate(fs):
        c = uvc[fr == f][:K]; c = c[np.argsort(-c[:, 2])]; S[i, :len(c)] = c
    X = [np.isfinite(S[:, :, 2]), np.nan_to_num(S[:, :, 2])]
    for dt in DTS:
        N = np.full_like(S, np.nan)                                                     # N[i] = S[i + dt]
        if dt > 0: N[:-dt] = S[dt:]
        else: N[-dt:] = S[:dt]
        d = np.linalg.norm(S[:, :, None, :2] - N[:, None, :, :2], axis=3); d = np.where(np.isfinite(d), d, np.inf)
        j = d.argmin(2); dm = np.take_along_axis(d, j[:, :, None], 2)[:, :, 0]; cn = np.take_along_axis(np.nan_to_num(N[:, :, 2])[:, None, :].repeat(K, 1), j[:, :, None], 2)[:, :, 0]
        X += [np.minimum(dm, CLIP) / CLIP, np.where(np.isfinite(dm), cn, 0)]
    X = np.stack(X, 2); X[~np.isfinite(S[:, :, 2])] = 0
    return fs, X.reshape(len(fs), -1).astype(np.float32), S

def net():
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(s, d=K * (2 + 2 * len(DTS)), h=48):
            super().__init__(); s.gru = nn.GRU(d, h, num_layers=2, batch_first=True, bidirectional=True); s.out = nn.Linear(2 * h, K + 1)
        def forward(s, x): return s.out(s.gru(x)[0])
    return Net()

def sequences(det, gt, clips):
    """(X, y, S) per clip; label = slot within max(3 px, ball px) of the truth, else K (none, incl. frames without truth)."""
    out = []
    for ci in clips:
        d = det[det[:, 0] == ci]; g = {int(r[1]): r[2:] for r in gt[gt[:, 0] == ci]}
        if not len(d): continue
        fs, X, S = features(d[:, 1].astype(int), d[:, 2:5]); y = np.full(len(fs), K)
        for i, f in enumerate(fs):
            if f in g:
                dd = np.hypot(S[i, :, 0] - g[f][0], S[i, :, 1] - g[f][1]); j = np.nanargmin(np.where(np.isfinite(dd), dd, np.inf))
                if dd[j] <= max(3, g[f][2]): y[i] = j
        out.append((X, y, S))
    return out

def fit(epochs=60, seed=0):
    import torch; torch.manual_seed(seed)
    z = np.load(CANDS); det, gt = z["det"], z["gt"]; n = int(det[:, 0].max()) + 1; from src.soccernet import VAL_CLIPS
    tr, va = sequences(det, gt, range(n - VAL_CLIPS)), sequences(det, gt, range(n - VAL_CLIPS, n))
    m = net(); opt = torch.optim.Adam(m.parameters(), 2e-3); lf = torch.nn.CrossEntropyLoss(); r = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        for i in r.permutation(len(tr)):
            X, y, _ = tr[i]; a = r.integers(0, max(1, len(y) - 200)); x = torch.tensor(X[a:a + 200])[None]   # 8 s windows
            opt.zero_grad(); loss = lf(m(x)[0], torch.tensor(y[a:a + 200])); loss.backward(); opt.step()
        if ep % 10 == 9: print(ep + 1, report(m, va), flush=True)
    torch.save(m.state_dict(), MODEL); print("->", MODEL)

def pick(m, X, thr=0.5):
    """Per frame: chosen slot or K, and its probability."""
    import torch
    with torch.no_grad(): p = torch.softmax(m(torch.tensor(X)[None])[0], 1).numpy()
    j = p[:, :K].argmax(1); pj = p[np.arange(len(j)), j]; return np.where(pj >= thr, j, K), pj

def report(m, va):
    """Held-out SoccerNet: of frames whose ball is among the candidates, share picked right; wrong picks; vs top-1 at conf 0.25."""
    m.eval(); ok = wrong = top_ok = top_wrong = n_ball = 0
    for X, y, S in va:
        j, _ = pick(m, X); n_ball += (y < K).sum(); ok += ((j == y) & (y < K)).sum(); wrong += ((j < K) & (j != y)).sum()
        t = np.where(np.nan_to_num(S[:, 0, 2]) >= 0.25, 0, K); top_ok += ((t == y) & (y < K)).sum(); top_wrong += ((t < K) & (t != y)).sum()
    return f"ball-in-candidates frames {n_ball}: GRU right {ok / n_ball:.0%}, wrong picks {wrong} | top-1 conf>=0.25 right {top_ok / n_ball:.0%}, wrong {top_wrong}"

def ours(path, thr=0.5):
    """Apply to our top-k file (rows f, x, y, conf, u, v), per run of consecutive frames; score as src/ball_track.py."""
    import torch; from src.ball_track import score, track
    m = net(); m.load_state_dict(torch.load(MODEL)); m.eval(); det = np.load(path)["det"]; fs = np.unique(det[:, 0].astype(int)); out = {}
    for run in np.split(fs, np.where(np.diff(fs) != 1)[0] + 1):
        d = det[np.isin(det[:, 0], run)]; f, X, S = features(d[:, 0].astype(int), d[:, [4, 5, 3]]); j, pj = pick(m, X, thr)
        for i in np.where(j < K)[0]:
            r = d[(d[:, 0] == f[i])]; r = r[np.argsort(-r[:, 3])][j[i]]; out[int(f[i])] = (r[1], r[2], pj[i])
    for g in (True, False):
        print("graded play only" if g else "all frames (incl. warm-up)")
        print(f"  GRU (p >= {thr})  :", score(out, graded=g)); print("  Viterbi tracked:", score(track(det), graded=g))

def selftest():
    """features(): a slot whose candidate moves smoothly gets small neighbour distances; a one-frame blip gets the cap."""
    fr = np.r_[np.arange(10), [5]]; uvc = np.c_[np.r_[np.arange(10) * 2.0, [300]], np.r_[np.full(10, 50.0), [300]], np.r_[np.full(10, 0.4), [0.9]]]
    fs, X, S = features(fr, uvc); x = X.reshape(len(fs), K, -1)
    assert S[5, 0, 0] == 300 and (x[5, 0, 2::2] == 1.0).all() and np.allclose(x[5, 1, 2::2], [0.1, 0.05, 0.05, 0.1]) and x[5, 1, 1] == np.float32(0.4), x[5]
    assert x[0, 0, 2] == 1 and x[0, 0, 3] == 0                      # no frame before the first: distance at the cap, no conf
    print("selftest ok")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cands", action="store_true"); ap.add_argument("--fit", action="store_true"); ap.add_argument("--ours")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--thr", type=float, default=0.5); a = ap.parse_args()
    cands() if a.cands else fit() if a.fit else ours(a.ours, a.thr) if a.ours else selftest()
