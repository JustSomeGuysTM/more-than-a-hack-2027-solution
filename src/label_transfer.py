"""Label transfer: learn DFL's official per-frame possession/ballstatus labels from *clean* tracking + the event stream
(a "teacher" HGB; DFL ball given BAR-ATH/PFF-like gaps and a moving dead ball so the teacher leans on the events), then label the PFF World Cup matches with it instead of the hand rule (which only reproduces the
official DFL labels at spellF1 0.73-0.76 when applied to DFL's own events, eda/NOTES.md 2026-10-28).

    .venv/bin/python src/label_transfer.py --cv       # teacher leave-one-match-out on DFL vs official labels
    .venv/bin/python src/label_transfer.py --train    # teacher on all 7 -> src/teacher.pkl
    .venv/bin/python src/label_transfer.py --label    # all PFF caches -> data_raw/external/pff_labels2_<gid>.npy (H/A/D)
"""
import argparse, glob, os, pickle, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.features import features
from src.possession import score, LAB
from src.v2 import IDS, fit, mirror, decode, V
from src.external import dead_ball_carry, degrade_ball
SW = 6   # teacher Viterbi switch: DFL LOMO grid 1/2/3/6/10 -> spellF1 0.737/0.747/0.751/0.756/0.756 vs official, macroF1 0.953 throughout; 6 keeps the spell count closest (393 vs 437)
PC = os.path.expanduser("~/pathcrf/data/sportec"); EXT = os.path.join(ROOT, "data_raw/external"); MODEL = os.path.join(ROOT, "src/teacher.pkl")
ONBALL = {"pass", "control", "interception", "clearance", "throw_in", "bad_touch", "tackle", "dispossessed", "cross", "shot", "shot_freekick", "freekick_short", "goalkick", "ball_recovery", "corner_short", "corner_crossed", "freekick_crossed", "keeper_claim", "keeper_pick_up", "keeper_punch", "keeper_save", "keeper_sweeper", "shot_block", "take_on", "dribble", "carry", "touch"}
DEAD = {"out", "foul", "goal", "offside", "whistle", "end", "substitution"}

def dfl(mid):
    """Clean DFL match dict + event instants: (frame, team 0/1) for on-ball events, frames of dead-type events, period starts."""
    z = np.load(f"{EXT}/dfl_{mid}.npz"); d = {q: z[q] for q in z.files}; d["fps"] = 25
    d = degrade_ball(dead_ball_carry(d, seed=hash(mid) % 1000), seed=hash(mid) % 1000)   # v1 (clean, frozen dead ball) labelled 0% of PFF frames dead: PFF's ball is missing/moving while dead
    tr = pd.read_parquet(f"{PC}/tracking_processed/{mid}.parquet", columns=["frame_id", "period_id"]); ev = pd.read_parquet(f"{PC}/event_processed/{mid}.parquet")
    pos = {f: i for i, f in enumerate(tr["frame_id"].values)}; on, dead = [], []
    for f, t, p in zip(ev["frame_id"], ev["event_type"], ev["player_id"]):
        if f not in pos: continue
        if t in ONBALL: on.append((pos[f], 0 if str(p).startswith("home") else 1))
        elif t in DEAD: dead.append(pos[f])
    d["events"] = (np.array(sorted(on)), np.array(sorted(dead)), tr.groupby("period_id")["frame_id"].first().map(pos).values); return d

def pff_events(z):
    """Same event representation from a PFF cache: on-ball instants = starts of otb_player runs; dead events = starts of rule-dead runs."""
    o = z["otb_player"]; L = z["labels"]; per = z["period"]; F = len(L)
    chg = np.r_[True, o[1:] != o[:-1]]; on = [(i, 0 if o[i].startswith("home") else 1) for i in np.flatnonzero(chg) if o[i]]
    ds = np.flatnonzero(np.r_[False, (L[1:] == "D") & (L[:-1] != "D")]); starts = np.r_[0, np.flatnonzero(per[1:] != per[:-1]) + 1]
    return np.array(on), ds, starts

def event_feats(F, events, cap=30.0):
    """Seconds since/until the nearest on-ball instant and its team, since/until dead-type events, since period start (capped)."""
    on, dead, starts = events; idx = np.arange(F); out = {}
    def since_until(fr):
        last = np.maximum.accumulate(np.where(np.isin(idx, fr), idx, -1)); nx = np.minimum.accumulate(np.where(np.isin(idx, fr), idx, F)[::-1])[::-1]
        return last, np.where(last >= 0, idx - last, 25 * cap).clip(0, 25 * cap) / 25.0, nx, np.where(nx < F, nx - idx, 25 * cap).clip(0, 25 * cap) / 25.0
    team = np.full(F, -1); team[on[:, 0]] = on[:, 1]; last, out["sinceOn"], nx, out["untilOn"] = since_until(on[:, 0])
    out["lastOnH"] = (np.where(last >= 0, team[last.clip(0)], -1) == 0).astype(float); out["lastOnA"] = (np.where(last >= 0, team[last.clip(0)], -1) == 1).astype(float)
    out["nextOnH"] = (np.where(nx < F, team[nx.clip(max=F - 1)], -1) == 0).astype(float); out["nextOnA"] = (np.where(nx < F, team[nx.clip(max=F - 1)], -1) == 1).astype(float)
    _, out["sinceDead"], _, out["untilDead"] = since_until(dead); _, out["sincePeriod"], _, _ = since_until(starts)
    names = list(out); return np.column_stack([out[k] for k in names]).astype(np.float32), names

def build(d):
    X, n = features(d, (105.0, 68.0)); E, en = event_feats(len(X), d["events"]); return np.hstack([X, E]), n + en

def mirror_all(X, names):
    Xm = mirror(X, names); sw = {"lastOnH": "lastOnA", "nextOnH": "nextOnA"}
    for a, b in sw.items(): Xm[:, names.index(a)], Xm[:, names.index(b)] = X[:, names.index(b)], X[:, names.index(a)]
    return Xm

def dataset(mids):
    Xs, ys = [], []
    for m in mids:
        d = dfl(m); X, names = build(d); y = (d["labels"] == "A") * 1 + (d["labels"] == "D") * 2; Xs += [X, mirror_all(X, names)]; ys += [y, np.where(y == 2, 2, 1 - y)]
    return np.concatenate(Xs), np.concatenate(ys), names

def cv():
    folds = []
    for held in IDS:
        X, y, names = dataset([m for m in IDS if m != held]); clf = fit(X, y); d = dfl(held); Xh, _ = build(d); lp = np.log(clf.predict_proba(Xh).clip(1e-4, 1))
        folds.append((d["labels"], lp)); print(held, "frame acc %.3f" % (LAB[lp.argmax(1)] == d["labels"]).mean(), flush=True)
    for sw in (1, 2, 3, 6, 10):
        r = [score(l, decode(lp, dict(V, switch=sw, switch_d=sw))) for l, lp in folds]
        print(f"teacher switch={sw:2d}: spellF1 {np.mean([x['spell_f1_mean'] for x in r]):.3f} macroF1 {np.mean([x['macro_f1'] for x in r]):.3f} spells {np.mean([x['n_pred_spells'] for x in r]):.0f} (gt {np.mean([x['n_gt_spells'] for x in r]):.0f})")

def train():
    X, y, names = dataset(IDS); clf = fit(X, y); pickle.dump({"clf": clf, "names": names}, open(MODEL, "wb")); print("teacher ->", MODEL)

def label():
    m = pickle.load(open(MODEL, "rb"))
    for f in sorted(glob.glob(f"{EXT}/pff_*.npz")):
        gid = os.path.basename(f)[4:-4]; z = np.load(f, allow_pickle=True); d = {q: z[q] for q in z.files}; d["fps"] = 25; d["events"] = pff_events(z)
        X, names = build(d); assert names == m["names"]; p = decode(np.log(m["clf"].predict_proba(X).clip(1e-4, 1)), dict(V, switch=SW, switch_d=SW)); np.save(f"{EXT}/pff_labels2_{gid}.npy", p)
        print(gid, "D frac %.2f (rule %.2f), label changes %d (rule %d)" % (np.mean(p == "D"), np.mean(z["labels"] == "D"), (p[1:] != p[:-1]).sum(), (z["labels"][1:] != z["labels"][:-1]).sum()), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cv", action="store_true"); ap.add_argument("--train", action="store_true"); ap.add_argument("--label", action="store_true"); a = ap.parse_args()
    cv() if a.cv else train() if a.train else label() if a.label else print(__doc__)
