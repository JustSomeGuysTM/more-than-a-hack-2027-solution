"""Possession v2: gradient-boosted per-frame classifier (trained on the 7 DFL open matches with the
ball degraded to BAR-ATH visibility) -> log P(H/A/D) as Viterbi emissions -> switch penalty.

    .venv/bin/python src/v2.py --cv          # leave-one-match-out on DFL: honest spellF1, picks temperature/switch
    .venv/bin/python src/v2.py --train       # fit on all 7 DFL matches -> src/v2_model.pkl
    .venv/bin/python src/v2.py               # BAR-ATH -> evaluation/results/v3.csv (+ public-window sanity score)

Training data is other games only (RULES.md); features are slot-agnostic (src/features.py); the
home/away mirror doubles the data and removes any home bias.
"""
import argparse, os, pickle, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.features import features
from src.possession import viterbi, score, LAB
from src.external import degrade_ball
IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
MODEL = os.path.join(ROOT, "src/v2_model.pkl")
VISIBLE = 0.57   # ATD ball visibility on graded frames; 0.78 after ball_fuse.py (detector fills gaps)
V = dict(temp=1.0, switch=10.0, switch_d=10.0)      # DFL leave-one-out: v2 spellF1 0.50 / macroF1 0.89; v3 (+since-touch features) 0.56 / 0.91 (2026-10-26). emissions are log-probs (~[-7, 0]); penalties are in the same units

def dfl_game(mid, seed):
    z = np.load(os.path.join(ROOT, f"data_raw/external/dfl_{mid}.npz")); d = {q: z[q] for q in z.files}; d["fps"] = 25
    return degrade_ball(d, visible=VISIBLE, seed=seed)

def mirror(X, names):
    """Swap home and away columns, negate the differences."""
    Xm = X.copy()
    for i, n in enumerate(names):
        if "H" in n and n.replace("H", "A") in names: Xm[:, i] = X[:, names.index(n.replace("H", "A"))]
        elif "A" in n and n.replace("A", "H") in names: Xm[:, i] = X[:, names.index(n.replace("A", "H"))]
        elif n.startswith("ddiff"): Xm[:, i] = -X[:, i]
    return Xm

def dataset(mids, seed=0):
    Xs, ys = [], []
    for k, mid in enumerate(mids):
        d = dfl_game(mid, seed + k); X, names = features(d, (105.0, 68.0)); y = (d["labels"] == "A") * 1 + (d["labels"] == "D") * 2  # H=0 A=1 D=2
        Xs += [X, mirror(X, names)]; ys += [y, np.where(y == 2, 2, 1 - y)]
    return np.concatenate(Xs), np.concatenate(ys), names

def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0).fit(X, y)

def decode(logp, v=V):
    return LAB[viterbi(v["temp"] * logp, v["switch"], v["switch_d"])]

def cv():
    folds = []
    for held in IDS:
        X, y, names = dataset([m for m in IDS if m != held]); clf = fit(X, y)
        d = dfl_game(held, 99); Xh, _ = features(d, (105.0, 68.0)); logp = np.log(clf.predict_proba(Xh).clip(1e-4, 1))
        folds.append((d["labels"], logp)); print(held, "frame acc %.3f" % (LAB[logp.argmax(1)] == d["labels"]).mean(), flush=True)
    for sw in (3, 6, 10, 15, 25):
        for t in (0.5, 1.0):
            r = [score(l, decode(lp, dict(temp=t, switch=sw, switch_d=sw))) for l, lp in folds]
            print(f"temp={t} switch={sw:2d}: spellF1={np.mean([x['spell_f1_mean'] for x in r]):.3f} macroF1={np.mean([x['macro_f1'] for x in r]):.3f} spells={np.mean([x['n_pred_spells'] for x in r]):.0f} (gt {np.mean([x['n_gt_spells'] for x in r]):.0f})")

def train():
    X, y, names = dataset(IDS); clf = fit(X, y); pickle.dump({"clf": clf, "names": names}, open(MODEL, "wb")); print("trained on", X.shape, "->", MODEL)

def run_barath(out, source="tactical"):
    import pandas as pd; from eda.atd import load; from games import GAMES, GAME
    m = pickle.load(open(MODEL, "rb")); d = load(source); d["fps"] = 25; F = len(d["ball"])
    X, names = features(d, GAMES[GAME]["pitch"]); assert names == m["names"]
    logp = np.log(m["clf"].predict_proba(X).clip(1e-4, 1))
    alive = np.zeros(F, bool); alive[11185:90652] = True; alive[113742:190198] = True; logp[~alive, :2] = -20   # outside the halves is dead by definition
    lab = decode(logp); pd.DataFrame({"frame": np.arange(F), "label": lab}).to_csv(out, index=False)
    gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
    print("BAR-ATH public window (sanity only):", score(gt, lab)); print("wrote", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cv", action="store_true"); ap.add_argument("--train", action="store_true"); ap.add_argument("--out", default=os.path.join(ROOT, "evaluation/results/v3.csv")); ap.add_argument("--source", default="tactical", help="tactical or tactical_fused (ball_fuse.py)"); ap.add_argument("--visible", type=float, default=VISIBLE, help="DFL ball visibility for --cv/--train; 0.78 to match tactical_fused")
    a = ap.parse_args(); VISIBLE = a.visible; cv() if a.cv else train() if a.train else run_barath(a.out, a.source)
