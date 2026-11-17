"""Ball detector for the tactical video, self-trained on this match's own ATD ball (algorithmic
pseudo-labels -> allowed: no person decides a label). Goal: a ball position on the 43% of frames
where the ATD has none.

    .venv/bin/python src/ball_detect.py --extract   # every 50th frame -> YOLO dataset under data_raw/ball/ (~12 min, one video decode)
    .venv/bin/python src/ball_detect.py --train     # YOLOv8n on 640x640 tiles (MPS, ~1 h)
    .venv/bin/python src/ball_detect.py --infer     # all frames -> eda/out/ball/detections.npz (frame, x_m, y_m, conf)
        [--up 2] [--weights coco|<pt>] [--step 5] [--topk 5 --window 100]   # tag in the file name; window = dense stretches
    .venv/bin/python src/ball_detect.py --check eda/out/ball/detections.npz --step 50 [--offset 25]   # vs the ATD ball

Pseudo-label = ATD ball projected through the frame's homography, then snapped to the brightest
small blob within SNAP px (the ATD position is a smoothed ground-plane estimate, ~5-20 px off).
Tiles: 640x640 crops; training tiles are centred near the ball with a random offset, plus one
ball-free tile per frame. Inference tiles the 1920x1080 frame 4x2 with overlap.
"""
import argparse, os, subprocess, sys
import numpy as np
from scipy.ndimage import gaussian_filter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from games import asset_paths, GAMES, GAME
from src.possession import play_windows
L, W = GAMES[GAME]["pitch"]; VW, VH = 1920, 1080; T = 640; BOX = 20; SNAP = 30; STEP = 50
DATA = os.path.join(ROOT, "data_raw/ball"); OUT = os.path.join(ROOT, "eda/out/ball"); WEIGHTS = os.path.join(ROOT, "src/ball_yolov8n.pt")

def homographies():
    return np.loadtxt(asset_paths(source="tactical")["homography"], delimiter=",", skiprows=1)[:, 1:10].reshape(-1, 3, 3)

def to_px(H, xy):
    v = H @ np.array([xy[0] / L, xy[1] / W, 1.0]); return v[0] / v[2], v[1] / v[2]

def to_pitch(H, u, v):
    p = np.linalg.inv(H) @ np.array([u, v, 1.0]); return p[0] / p[2] * L, p[1] / p[2] * W

def frames(path, step=1, start=0, n=None):
    """Yield (frame_index, HxWx3 uint8) for every `step`-th frame, one sequential decode."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=not(mod(n\\,{step}))", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=VW * VH * 3 * 4); k = 0
    while True:
        buf = proc.stdout.read(VW * VH * 3)
        if len(buf) < VW * VH * 3: break
        yield k * step, np.frombuffer(buf, np.uint8).reshape(VH, VW, 3); k += 1
    proc.stdout.close(); proc.wait()

def snap(img, u, v):
    """Move (u,v) to the brightest small blob nearby: DoG peak on the grey crop."""
    u0, v0 = int(round(u)), int(round(v)); r = SNAP + 8
    if not (r <= u0 < VW - r and r <= v0 < VH - r): return u, v, 0.0
    g = img[v0 - r:v0 + r, u0 - r:u0 + r].astype(np.float32).mean(2)
    dog = gaussian_filter(g, 1.5) - gaussian_filter(g, 4.0); yy, xx = np.mgrid[-r:r, -r:r]; dog[(xx ** 2 + yy ** 2) > SNAP ** 2] = -1e9
    j, i = np.unravel_index(dog.argmax(), dog.shape); return u0 + i - r, v0 + j - r, float(dog.max())

def extract():
    from PIL import Image; from eda.atd import load
    d = load("tactical"); H = homographies(); rng = np.random.default_rng(0); n_pos = n_neg = 0; moved = []
    for split in ("train", "val"):
        for s in ("images", "labels"): os.makedirs(f"{DATA}/{split}/{s}", exist_ok=True)
    for f, img in frames(asset_paths(source="tactical")["video"], STEP):
        if not np.isfinite(H[f]).all(): continue
        split = "val" if (f // STEP) % 10 == 0 else "train"; b = d["ball"][f]
        if np.isfinite(b[0]):
            u, v = to_px(H[f], b); u2, v2, c = snap(img, u, v)
            if c > 0: moved.append(np.hypot(u2 - u, v2 - v)); u, v = u2, v2
            x0 = int(np.clip(u - rng.integers(40, T - 40), 0, VW - T)); y0 = int(np.clip(v - rng.integers(40, T - 40), 0, VH - T))
            Image.fromarray(img[y0:y0 + T, x0:x0 + T]).save(f"{DATA}/{split}/images/f{f}_pos.jpg", quality=92)
            open(f"{DATA}/{split}/labels/f{f}_pos.txt", "w").write(f"0 {(u - x0) / T:.5f} {(v - y0) / T:.5f} {BOX / T:.5f} {BOX / T:.5f}\n"); n_pos += 1
            far = lambda x, y: abs(x + T / 2 - u) > T * 0.6 or abs(y + T / 2 - v) > T * 0.6
        else: far = lambda x, y: True
        for _ in range(4):                                                    # one ball-free tile per frame
            x0, y0 = int(rng.integers(0, VW - T)), int(rng.integers(0, VH - T))
            if far(x0, y0):
                Image.fromarray(img[y0:y0 + T, x0:x0 + T]).save(f"{DATA}/{split}/images/f{f}_neg.jpg", quality=92); open(f"{DATA}/{split}/labels/f{f}_neg.txt", "w").close(); n_neg += 1; break
        if f % 5000 == 0: print(f, n_pos, n_neg, flush=True)
    open(f"{DATA}/ball.yaml", "w").write(f"path: {DATA}\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n")
    print("tiles: pos", n_pos, "neg", n_neg, "| snap moved labels by median %.1f px, p90 %.1f px" % (np.median(moved), np.percentile(moved, 90)))

def train(epochs):
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt"); m.train(data=f"{DATA}/ball.yaml", imgsz=T, epochs=epochs, batch=32, device="mps", project=f"{DATA}/runs", name="ball", exist_ok=True, mosaic=0.5, scale=0.2, fliplr=0.5, pretrained=True, verbose=False)
    import shutil; shutil.copy(f"{DATA}/runs/ball/weights/best.pt", WEIGHTS); print("->", WEIGHTS)

def infer(step, up=1.0, weights=WEIGHTS, topk=1, window=0):
    """Top-k detections per frame (rows sorted by confidence; candidates within 10 px merged). up > 1 upsamples the frame
    first (see eda/NOTES.md "SoccerNet ball data, day one"); tiles of 640 with >= 20% overlap. weights=coco: stock yolov8n,
    sports ball. window=N: only frames with f % 500 < N (dense stretches for src/ball_track.py), decoded at step 1."""
    import cv2; from ultralytics import YOLO
    coco = weights == "coco"; m = YOLO("yolov8n.pt" if coco else weights); H = homographies(); rows = []
    W_, H_ = int(VW * up), int(VH * up); grid = lambda n: np.linspace(0, n - T, int(np.ceil((n - T) / (T * 0.8))) + 1).astype(int)
    xs, ys = grid(W_), grid(H_)                                                   # up=1: the original 4x2 tiles (0/427/853/1280, 0/440)
    for f, img in frames(asset_paths(source="tactical")["video"], 1 if window else step):
        if not np.isfinite(H[f]).all() or (window and f % 500 >= window): continue
        img = np.ascontiguousarray(img[..., ::-1])   # ffmpeg gives RGB; ultralytics treats arrays as BGR (as it reads the training jpgs)
        if up != 1: img = cv2.resize(img, (W_, H_), interpolation=cv2.INTER_CUBIC)
        tiles = [img[y:y + T, x:x + T] for y in ys for x in xs]
        res = m.predict(tiles, imgsz=T, conf=0.1, device="mps", verbose=False, classes=[32] if coco else None); cand = []
        for (x, y), r in zip([(x, y) for y in ys for x in xs], res):
            cand += [((x + bx[0]) / up, (y + bx[1]) / up, float(cf)) for bx, cf in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        kept = []
        for u, v, cf in sorted(cand, key=lambda c: -c[2]):
            if len(kept) < topk and all(np.hypot(u - a, v - b) > 10 for a, b, _ in kept): kept.append((u, v, cf))
        for u, v, cf in kept: px, py = to_pitch(H[f], u, v); rows.append((f, px, py, cf, u, v))
        if f % 5000 == 0: print(f, len(rows), flush=True)
    tag = ("_coco" if coco else "" if weights == WEIGHTS else "_" + os.path.basename(weights)[:-3].replace("ball_yolov8n_", "")) + (f"_up{up:g}" if up != 1 else "") + \
          (f"_top{topk}" if topk > 1 else "") + (f"_win{window}" if window else "" if step == 1 else "_step" + str(step))
    os.makedirs(OUT, exist_ok=True); np.savez_compressed(f"{OUT}/detections{tag}.npz", det=np.array(rows, np.float32)); print("detections", len(rows), tag)

def check(path, step, offset=0, window=0, graded=False):
    """Detections vs the ATD ball on every step-th frame (from offset) with a homography (the NOTES "Ball detector" table).
    Training tiles come from frames f % 50 == 0: score a detector trained on them with --step 50 --offset 25."""
    from eda.atd import load
    atd = load("tactical")["ball"]; H = homographies(); det = np.load(path)["det"]; det = det[det[:, 0] % step == offset]
    fr = np.array([f for f in range(offset, len(atd), step) if np.isfinite(H[f]).all() and (not window or f % 500 < window)
                   and (not graded or any(a <= f < b for a, b in play_windows()))])   # graded play, as src/ball_fuse.py
    seen = np.isfinite(atd[fr, 0]); by = {int(r[0]): r for r in det[::-1]}                     # top-k files: keep each frame's best
    print(f"{os.path.basename(path)}: {len(fr)} frames, ATD ball on {seen.mean():.0%}")
    for c in (0.1, 0.25, 0.4, 0.6):
        hit = np.array([f in by and by[f][3] >= c for f in fr]); err = np.array([np.hypot(*(by[f][1:3] - atd[f, :2])) for f in fr[seen & hit]])
        print(f"  conf {c:.2f}: detects on ATD-ball frames {hit[seen].mean():.0%}, median error {np.median(err):.1f} m, >5 m {np.mean(err > 5):.0%} | detects on ATD-missing frames {hit[~seen].mean():.0%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--train", action="store_true"); ap.add_argument("--infer", action="store_true"); ap.add_argument("--check")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--up", type=float, default=1.0); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--topk", type=int, default=1); ap.add_argument("--window", type=int, default=0); ap.add_argument("--graded", action="store_true"); ap.add_argument("--weights", default=WEIGHTS); a = ap.parse_args()
    extract() if a.extract else train(a.epochs) if a.train else check(a.check, a.step, a.offset, a.window, a.graded) if a.check else infer(a.step, a.up, a.weights, a.topk, a.window)
