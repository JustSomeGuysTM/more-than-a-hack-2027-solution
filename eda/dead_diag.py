"""Where do dead-ball errors cost spellF1? HGB v3 leave-one-match-out folds on DFL (cached), then:
oracle-D upper bound, D onset/offset lag, missed stoppages by duration, switch_d grid.
    .venv/bin/python eda/dead_diag.py          # folds cached in eda/out/dead_folds.npz
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.v2 import IDS, dataset, fit, dfl_game, decode, V
from src.features import features
from src.possession import score, LAB
CACHE = os.path.join(ROOT, f"eda/out/dead_folds{os.environ.get('TAG', '')}.npz")

def folds():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True); return list(zip(z["labels"], z["logp"]))
    out = []
    for held in IDS:
        X, y, _ = dataset([m for m in IDS if m != held]); clf = fit(X, y); d = dfl_game(held, 99); Xh, _ = features(d, (105.0, 68.0))
        out.append((d["labels"], np.log(clf.predict_proba(Xh).clip(1e-4, 1)))); print(held, flush=True)
    np.savez(CACHE, labels=np.array([l for l, _ in out], dtype=object), logp=np.array([lp for _, lp in out], dtype=object)); return out

def runs(mask):
    """(start, end) of True runs."""
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def oracle_d(lp, gt):
    o = lp.copy(); dead = gt == "D"; o[dead, :2] = -20; o[dead, 2] = 0; o[~dead, 2] = -20; return o

def main():
    F = folds(); fps = 25
    for name, f in (("own D", lambda lp, gt: decode(lp)), ("oracle D", lambda lp, gt: decode(oracle_d(lp, gt)))):
        r = [score(gt, f(lp, gt)) for gt, lp in F]; print(f"{name:9s} spellF1 {np.mean([x['spell_f1_mean'] for x in r]):.3f} macroF1 {np.mean([x['macro_f1'] for x in r]):.3f}")
    print("switch_d grid (switch 10):", {sd: round(float(np.mean([score(gt, decode(lp, dict(V, switch_d=sd)))["spell_f1_mean"] for gt, lp in F])), 3) for sd in (2, 4, 6, 10, 15, 25)})
    on, off, missed, false, durs = [], [], [], 0, []
    for gt, lp in F:
        pd_ = decode(lp) == "D"; gd = gt == "D"; ps = runs(pd_)
        for s, e in runs(gd):
            durs.append((e - s) / fps); ov = [(a, b) for a, b in ps if b > s and a < e]
            if not ov: missed.append((e - s) / fps); continue
            on.append((ov[0][0] - s) / fps); off.append((ov[-1][1] - e) / fps)
        false += sum(1 for a, b in ps if not gd[a:b].any())
    q = lambda v: np.round(np.percentile(v, [10, 50, 90]), 2)
    print(f"GT stoppages {len(durs)}, median {np.median(durs):.1f} s | missed {len(missed)} (median dur {np.median(missed) if missed else 0:.1f} s) | false D runs {false}")
    print("onset lag s (p10/50/90, + = late):", q(on), "| offset lag s:", q(off))
    print("missed by duration s:", {b: sum(1 for m in missed if lo <= m < hi) for b, (lo, hi) in {"<3": (0, 3), "3-6": (3, 6), "6-12": (6, 12), ">12": (12, 1e9)}.items()},
          "| GT by duration:", {b: sum(1 for m in durs if lo <= m < hi) for b, (lo, hi) in {"<3": (0, 3), "3-6": (3, 6), "6-12": (6, 12), ">12": (12, 1e9)}.items()})

def public(csv):
    """Dead-ball stats of a BAR-ATH result on the public window (sanity only, never tune on it)."""
    import pandas as pd; w = slice(11185, 18685); gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values[w]; gd = gt == "D"
    pdd = pd.read_csv(csv)["label"].values[w] == "D"; ps = runs(pdd); on, off, miss = [], [], 0
    for s, e in runs(gd):
        ov = [(a, b) for a, b in ps if b > s and a < e]
        if not ov: miss += 1; continue
        on.append((ov[0][0] - s) / 25); off.append((ov[-1][1] - e) / 25)
    print(f"{os.path.basename(csv):16s} D frames {pdd.sum()} (gt {gd.sum()}) runs {len(ps)} (gt {len(runs(gd))}) missed {miss} false {sum(1 for a, b in ps if not gd[a:b].any())} onset lag p50 {np.median(on):+.2f} s offset p50 {np.median(off):+.2f} s")

if __name__ == "__main__": public(sys.argv[1]) if len(sys.argv) > 1 else main()
