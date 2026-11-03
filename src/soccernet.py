"""SoccerNet Tracking 2023 ball boxes (public, other games, human labels) for the ball detector
(plans/soccernet_ball_detector.md). Unzip outside the repo first:
    unzip -q data_raw/external/soccernet/SN-Tracking-2023/train.zip -d ~/sn     # 8.9 GB; add "train/*/gt/gt.txt" "train/*/*.ini" for labels only, "train/SNMOT-06[0-4]/img1/*" for --zeroshot
    .venv/bin/python src/soccernet.py [--ours]         # ball coverage and box-size percentiles; --ours adds our tactical feed (local only)
    .venv/bin/python src/soccernet.py --zeroshot       # stock COCO yolov8n "sports ball" on clips with images, at SoccerNet and at our scale
"""
import argparse, configparser, glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
SN = os.path.expanduser(os.environ.get("SN", "~/sn/train"))
BALL_M = 0.22   # ball diameter

def ball_boxes(clip):
    """(frame, x, y, w, h) of the clip's ball track(s) in pixels; the ball's track id comes from gameinfo.ini."""
    c = configparser.ConfigParser(); c.read(os.path.join(clip, "gameinfo.ini"))
    ids = [int(k.split("_")[1]) for k, v in c["Sequence"].items() if k.startswith("trackletid_") and v.strip().startswith("ball")]
    gt = np.loadtxt(os.path.join(clip, "gt/gt.txt"), delimiter=",", ndmin=2)
    return gt[np.isin(gt[:, 1], ids)][:, [0, 2, 3, 4, 5]]

def ours():
    """Ball diameter in px on our tactical feed at our detector's confident hits: 0.22 m through the frame's homography."""
    from src.ball_detect import homographies, to_px
    H = homographies(); d = np.load(os.path.join(ROOT, "eda/out/ball/detections.npz"))["det"]; d = d[(d[:, 3] > 0.5) & (d[:, 0] % 25 == 0)]
    out = []
    for f, x, y, *_ in d:
        h = H[int(f)]; u0, v0 = to_px(h, (x, y)); u1, v1 = to_px(h, (x + BALL_M, y)); u2, v2 = to_px(h, (x, y + BALL_M))
        out.append((np.hypot(u1 - u0, v1 - v0) + np.hypot(u2 - u0, v2 - v0)) / 2)
    return np.array(out)

def zeroshot(runs=((1.0, 1), (0.3, 1), (0.3, 2.2)), step=15, conf=0.1):
    """Baseline every detector must beat: stock COCO yolov8n, class 32, top-1 box per frame. Hit = within max(8 px, box size)
    of the labelled ball centre (both in original pixels). Each run = (scale, upsample): shrink the frame to our tactical
    ball size (0.3), then let YOLO upsample it by `up` (imgsz = width * up), the way tiled inference would. Saves 6 frames with GT (green) and top-1 (red) for a look."""
    import cv2; from ultralytics import YOLO
    w = os.path.join(ROOT, "yolov8n.pt"); m = YOLO(w if os.path.exists(w) else "yolov8n.pt")   # gitignored; ultralytics downloads it by name
    clips = [c for c in sorted(glob.glob(os.path.join(SN, "SNMOT-*"))) if os.path.isdir(os.path.join(c, "img1"))]
    os.makedirs(os.path.join(ROOT, "eda/out/soccernet"), exist_ok=True); shown = 0
    for sc, up in runs:
        hit = miss = fp = n = 0
        for c in clips:
            b = ball_boxes(c); g = {int(f): (x + w / 2, y + h / 2, max(8, np.sqrt(w * h))) for f, x, y, w, h in b}
            for f in range(1, 751, step):
                img = cv2.imread(os.path.join(c, "img1", f"{f:06d}.jpg")); im = cv2.resize(img, None, fx=sc, fy=sc) if sc != 1 else img
                r = m.predict(im, imgsz=-(-int(im.shape[1] * up) // 32) * 32, conf=conf, classes=[32], device="mps", verbose=False)[0]; n += 1
                top = r.boxes.xywh.cpu().numpy()[r.boxes.conf.cpu().numpy().argmax()] / sc if len(r.boxes) else None
                if f in g and top is not None and np.hypot(top[0] - g[f][0], top[1] - g[f][1]) <= g[f][2]: hit += 1
                else: miss += f in g; fp += top is not None
                if sc == 1 and up == 1 and shown < 6 and f % 150 == 1:
                    if f in g: cv2.circle(img, (int(g[f][0]), int(g[f][1])), 20, (0, 255, 0), 2)
                    if top is not None: cv2.circle(img, (int(top[0]), int(top[1])), 30, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(ROOT, f"eda/out/soccernet/{os.path.basename(c)}_{f}.jpg"), img); shown += 1
        lab = hit + miss
        print(f"scale {sc} x{up}: {n} frames, ball labelled {lab}; recall {hit / max(lab, 1):.2f}, wrong or no-ball top-1 {fp} ({fp / n:.0%} of frames)")

def main(a):
    clips = sorted(glob.glob(os.path.join(SN, "SNMOT-*"))); q = [5, 25, 50, 75, 95]; cov, sizes = [], []
    for c in clips:
        s = configparser.ConfigParser(); s.read(os.path.join(c, "seqinfo.ini")); n = int(s["Sequence"]["seqLength"])
        b = ball_boxes(c); cov.append(len(np.unique(b[:, 0])) / n); sizes.append(np.sqrt(b[:, 3] * b[:, 4]))
    s = np.concatenate(sizes)
    print(f"SoccerNet: {len(clips)} clips, {len(s)} ball boxes, ball labelled on {np.mean(cov):.0%} of frames (clip p10/p50/p90 {np.round(np.percentile(cov, [10, 50, 90]), 2)})")
    print(f"  ball size px (sqrt w*h) p5/25/50/75/95: {np.round(np.percentile(s, q), 1)}")
    if a.zeroshot: return zeroshot()
    if a.ours:
        o = ours(); print(f"ours (tactical, {len(o)} confident hits): ball diameter px p5/25/50/75/95: {np.round(np.percentile(o, q), 1)}")
        print(f"  scale SoccerNet frames by ~{np.median(o) / np.median(s):.2f} (median) to match")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ours", action="store_true"); ap.add_argument("--zeroshot", action="store_true"); main(ap.parse_args())
