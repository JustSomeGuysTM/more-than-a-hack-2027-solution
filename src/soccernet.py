"""SoccerNet Tracking 2023 ball boxes (public, other games, human labels) for the ball detector
(plans/soccernet_ball_detector.md). Unzip outside the repo first:
    unzip -q data_raw/external/soccernet/SN-Tracking-2023/train.zip -d ~/sn     # 8.9 GB; add "train/*/gt/gt.txt" "train/*/*.ini" for labels only, "train/SNMOT-06[0-4]/img1/*" for --zeroshot
    .venv/bin/python src/soccernet.py [--ours]         # ball coverage and box-size percentiles; --ours adds our tactical feed (local only)
    .venv/bin/python src/soccernet.py --zeroshot       # stock COCO yolov8n "sports ball" on clips with images, at SoccerNet and at our scale
    .venv/bin/python src/soccernet.py --build          # YOLO set at the x2 inference scale -> data_raw/ball_up2/ (needs all images unzipped)
    .venv/bin/python src/soccernet.py --train          # YOLOv8n from COCO on it -> src/ball_yolov8n_up2.pt (bucket only: it saw our match's tiles)
        --up 1 on both: native scale (data_raw/ball_sn -> src/ball_yolov8n_sn.pt)
        --up 1 --box 20 [--init src/ball_yolov8n.pt]: 20 px label boxes, fine-tuned from our weights -> src/ball_yolov8n_sn20ft.pt
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

UP = 2.0; T = 640; OUR_PX = (3.0, 7.0)   # inference upsampling; our ball diameter p5-p95 on the tactical feed (px, native)
UP2 = os.path.join(ROOT, "data_raw/ball_up2"); VAL_CLIPS = 10   # last 10 SoccerNet clips held out

data_dir = lambda up, box=0: UP2 if up != 1 else os.path.join(ROOT, "data_raw/ball_sn" + (f"{box:g}" if box else ""))

def _save(D, split, name, img, box=None):
    import cv2
    cv2.imwrite(os.path.join(D, split, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    open(os.path.join(D, split, "labels", name + ".txt"), "w").write("" if box is None else "0 %.5f %.5f %.5f %.5f\n" % box)

def _crop(r, img, u=None, v=None):
    """A T x T crop (smaller if the image is), holding (u, v) at a random spot >= 40 px from the edge; random if no point."""
    h, w = img.shape[:2]; tw, th = min(T, w), min(T, h)
    x0 = int(np.clip(u - r.integers(40, max(41, tw - 40)), 0, w - tw)) if u is not None else int(r.integers(0, w - tw + 1))
    y0 = int(np.clip(v - r.integers(40, max(41, th - 40)), 0, h - th)) if v is not None else int(r.integers(0, h - th + 1))
    return img[y0:y0 + th, x0:x0 + tw], x0, y0

def build(step=5, up=UP, box=0):
    """SoccerNet frames shrunk so the ball has our native size (random over OUR_PX), softened, then upsampled x UP exactly as
    `src/ball_detect.py --infer --up` does; one ball tile + one ball-free tile per frame. Plus our own ATD tiles
    (data_raw/ball, algorithmic labels) upsampled x UP. Labels are boxes in the crop, YOLO format. up=1: the native set
    (data_raw/ball_sn), inference without upsampling. box: label box side in px (0 = ball size, min 6 px; 20 = the convention
    the current weights were trained with: context for the stride-8 head) -> data_raw/ball_sn{box}."""
    import cv2, shutil; r = np.random.default_rng(0); D = data_dir(up, box); shutil.rmtree(D, ignore_errors=True); n = {}
    for sp in ("train", "val"):
        for q in ("images", "labels"): os.makedirs(os.path.join(D, sp, q))
    clips = sorted(glob.glob(os.path.join(SN, "SNMOT-*")))
    for ci, c in enumerate(clips):
        sp = "val" if ci >= len(clips) - VAL_CLIPS else "train"; b = {int(f): (x, y, w, h) for f, x, y, w, h in ball_boxes(c)}
        for f in range(1, 751, step):
            if f not in b: continue
            x, y, w, h = b[f]; size = np.sqrt(w * h); s = r.uniform(*OUR_PX) / max(size, 1)
            img = cv2.imread(os.path.join(c, "img1", f"{f:06d}.jpg")); small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), r.uniform(0.3, 0.8))                       # our feed is softer than broadcast
            big = cv2.resize(small, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1 else small; k = s * up
            u, v, bw = (x + w / 2) * k, (y + h / 2) * k, max(size * k, box or 6)
            tile, x0, y0 = _crop(r, big, u, v); th, tw = tile.shape[:2]
            _save(D, sp, f"sn{ci:02d}_{f}_pos", tile, ((u - x0) / tw, (v - y0) / th, bw / tw, bw / th)); n[sp, "sn_pos"] = n.get((sp, "sn_pos"), 0) + 1
            for _ in range(5):
                tile, x0, y0 = _crop(r, big)
                if not (x0 - 20 < u < x0 + tile.shape[1] + 20 and y0 - 20 < v < y0 + tile.shape[0] + 20):
                    _save(D, sp, f"sn{ci:02d}_{f}_neg", tile); n[sp, "sn_neg"] = n.get((sp, "sn_neg"), 0) + 1; break
        print(os.path.basename(c), sp, flush=True)
    for sp in ("train", "val"):                                                                  # our ATD tiles, same split as data_raw/ball
        for p in sorted(glob.glob(os.path.join(ROOT, "data_raw/ball", sp, "images", "*.jpg"))):
            img = cv2.imread(p); img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1 else img; lab = open(p.replace("/images/", "/labels/")[:-4] + ".txt").read().split()
            name = "our_" + os.path.basename(p)[:-4]
            if lab:
                u, v = float(lab[1]) * T * up, float(lab[2]) * T * up; tile, x0, y0 = _crop(r, img, u, v); bw = (box or 6) * up   # ball ~4 px native + snap slack
                _save(D, sp, name, tile, ((u - x0) / T, (v - y0) / T, bw / T, bw / T)); n[sp, "our_pos"] = n.get((sp, "our_pos"), 0) + 1
            else: _save(D, sp, name, _crop(r, img)[0]); n[sp, "our_neg"] = n.get((sp, "our_neg"), 0) + 1
    open(os.path.join(D, "ball.yaml"), "w").write(f"path: {D}\ntrain: train/images\nval: val/images\nnames:\n  0: ball\n"); print(n)

def train(epochs=25, up=UP, box=0, init="yolov8n.pt"):
    """init: COCO, or a path to fine-tune existing weights (e.g. src/ball_yolov8n.pt)."""
    from ultralytics import YOLO; import shutil; D = data_dir(up, box)
    m = YOLO(init); m.train(data=os.path.join(D, "ball.yaml"), imgsz=T, epochs=epochs, batch=32, device="mps", project=os.path.join(D, "runs"), name="ball", exist_ok=True,
                                     mosaic=0.5, scale=0.2, fliplr=0.5, pretrained=True, verbose=False)
    tag = 'up2' if up != 1 else f"sn{box or ''}" + ('ft' if init != "yolov8n.pt" else ''); out = os.path.join(ROOT, f"src/ball_yolov8n_{tag}.pt"); shutil.copy(os.path.join(D, "runs/ball/weights/best.pt"), out); print("->", out)

def main(a):
    clips = sorted(glob.glob(os.path.join(SN, "SNMOT-*"))); q = [5, 25, 50, 75, 95]; cov, sizes = [], []
    for c in clips:
        s = configparser.ConfigParser(); s.read(os.path.join(c, "seqinfo.ini")); n = int(s["Sequence"]["seqLength"])
        b = ball_boxes(c); cov.append(len(np.unique(b[:, 0])) / n); sizes.append(np.sqrt(b[:, 3] * b[:, 4]))
    s = np.concatenate(sizes)
    print(f"SoccerNet: {len(clips)} clips, {len(s)} ball boxes, ball labelled on {np.mean(cov):.0%} of frames (clip p10/p50/p90 {np.round(np.percentile(cov, [10, 50, 90]), 2)})")
    print(f"  ball size px (sqrt w*h) p5/25/50/75/95: {np.round(np.percentile(s, q), 1)}")
    if a.zeroshot: return zeroshot()
    if a.build: return build(up=a.up, box=a.box)
    if a.train: return train(a.epochs, a.up, a.box, a.init)
    if a.ours:
        o = ours(); print(f"ours (tactical, {len(o)} confident hits): ball diameter px p5/25/50/75/95: {np.round(np.percentile(o, q), 1)}")
        print(f"  scale SoccerNet frames by ~{np.median(o) / np.median(s):.2f} (median) to match")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ours", action="store_true"); ap.add_argument("--zeroshot", action="store_true"); ap.add_argument("--build", action="store_true"); ap.add_argument("--train", action="store_true"); ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--up", type=float, default=UP); ap.add_argument("--box", type=float, default=0); ap.add_argument("--init", default="yolov8n.pt"); main(ap.parse_args())
