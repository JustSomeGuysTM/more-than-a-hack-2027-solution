"""Possession v1: ball gap filling -> per-frame evidence for H / A / D -> Viterbi over the three
states with a switch penalty. numpy only.

    .venv/bin/python src/possession.py                 # BAR-ATH -> evaluation/results/v1.csv, scored on the public 5 min
    .venv/bin/python src/possession.py --external      # honest numbers on Metrica games (labels derived from their events)

Evidence (log-scores per frame):
  H/A : (CAP - distance from ball to nearest home/away player) / SIGMA, >= 0; 0 when the ball is unknown
  D   : DEAD_OFF if the ball is off the pitch or before kickoff / after the half;
        DEAD_MISS if the ball is untracked (after short-gap interpolation);
        DEAD_STILL if the ball sits > STILL_DIST m from everyone and moves < STILL_SPEED m/s
  Viterbi: staying costs 0, switching costs SWITCH (log-odds), D<->team switches cost SWITCH_D.
Constants are set by eye from EDA and should be tuned on external games (src/external.py), never
on the public window.
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
P = dict(gap=50, sigma=1.0, cap=8.0, dead_off=6.0, dead_miss=0.6, dead_still=3.0, still_dist=4.0, still_speed=0.6, switch=60.0, switch_d=42.0, margin=0.5)   # tuned on Metrica 1+2 with degraded ball (src/external.py), 2026-10-26
LAB = np.array(list("HAD"))

def fill_ball(b, max_gap):
    """Linear interpolation of gaps <= max_gap frames; longer gaps stay NaN."""
    b = b.copy(); ok = np.isfinite(b[:, 0]); idx = np.flatnonzero(ok)
    if len(idx) < 2: return b
    out = np.column_stack([np.interp(np.arange(len(b)), idx, b[ok, k]) for k in range(2)])
    gaps = np.diff(idx); bad = np.zeros(len(b), bool)
    for s, g in zip(idx[:-1][gaps > max_gap + 1], gaps[gaps > max_gap + 1]): bad[s + 1:s + g] = True
    bad[:idx[0]] = bad[idx[-1] + 1:] = True
    out[bad] = np.nan; return out

def evidence(d, pitch, alive_mask, p=P):
    pl, team = d["players"][:, :, :2], d["team"]; L, W = pitch; fps = d.get("fps", 25)
    b = fill_ball(d["ball"][:, :2], p["gap"]); known = np.isfinite(b[:, 0])
    dist = np.linalg.norm(pl - b[:, None, :], axis=2)                                   # (F,N), NaN where either unknown
    dH = np.nanmin(np.where(team == 0, dist, np.nan), axis=1, initial=np.inf); dA = np.nanmin(np.where(team == 1, dist, np.nan), axis=1, initial=np.inf)
    e = np.zeros((len(b), 3))
    e[:, 0] = np.where(known, (p["cap"] - np.minimum(dH, p["cap"])) / p["sigma"], 0.0); e[:, 1] = np.where(known, (p["cap"] - np.minimum(dA, p["cap"])) / p["sigma"], 0.0)
    off = known & ((b[:, 0] < -p["margin"]) | (b[:, 0] > L + p["margin"]) | (b[:, 1] < -p["margin"]) | (b[:, 1] > W + p["margin"]))
    speed = np.r_[0, np.linalg.norm(np.diff(b, axis=0), axis=1)] * fps
    still = known & (np.minimum(dH, dA) > p["still_dist"]) & (speed < p["still_speed"])
    e[:, 2] = p["dead_off"] * (off | ~alive_mask) + p["dead_miss"] * ~known + p["dead_still"] * still
    e[:, :2] -= p["dead_off"] * ~alive_mask[:, None]
    return e

def viterbi(e, switch, switch_d):
    """Max-score path over 3 states. trans[i,j] = cost of going i->j."""
    F = len(e); trans = np.full((3, 3), -switch); trans[2, :2] = trans[:2, 2] = -switch_d; np.fill_diagonal(trans, 0)
    score = e[0].copy(); back = np.zeros((F, 3), np.int8)
    for t in range(1, F):
        cand = score[:, None] + trans; back[t] = cand.argmax(0); score = cand.max(0) + e[t]
    path = np.empty(F, np.int8); path[-1] = score.argmax()
    for t in range(F - 1, 0, -1): path[t - 1] = back[t, path[t]]
    return path

def predict(d, pitch, alive_mask=None, p=P):
    if alive_mask is None: alive_mask = np.ones(len(d["ball"]), bool)
    return LAB[viterbi(evidence(d, pitch, alive_mask, p), p["switch"], p["switch_d"])]

def score(gt_labels, pred_labels):
    sys.path.insert(0, os.path.join(ROOT, "evaluation")); import possession_metrics as pm
    r = pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(pred_labels))
    return {k: round(float(r[k]), 4) for k in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "n_gt_spells", "n_pred_spells", "gt_mean_spell_duration_s", "mean_spell_duration_s")}

def run_barath(out):
    from eda.atd import load; from games import GAMES, GAME
    d = load("tactical"); F = len(d["ball"]); alive = np.zeros(F, bool); alive[11185:90652] = True; alive[113742:190198] = True
    lab = predict(d, GAMES[GAME]["pitch"], alive)
    import pandas as pd
    pd.DataFrame({"frame": np.arange(F), "label": lab}).to_csv(out, index=False)
    gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
    print("BAR-ATH public window (sanity only, not for tuning):", score(gt, lab)); print("wrote", out)

def run_external():
    from src.external import metrica, degrade_ball
    for g in (1, 2):
        d = metrica(g); alive = np.ones(len(d["ball"]), bool)
        for name, dd in (("full ball", d), ("ball degraded", degrade_ball(d))):
            print(f"metrica {g} {name}:", score(d["labels"], predict(dd, (105.0, 68.0), alive)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--external", action="store_true"); ap.add_argument("--out", default=os.path.join(ROOT, "evaluation/results/v1.csv"))
    a = ap.parse_args()
    assert (viterbi(np.array([[1, 0, 0], [0, 1.5, 0], [1, 0, 0]], float), 2.0, 2.0) == 0).all()   # one blip is not worth a switch
    run_external() if a.external else run_barath(a.out)
