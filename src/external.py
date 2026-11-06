"""Other games in the same cache format as eda.atd.load(), plus derived H/A/D labels, for training
and honest validation (the competition's own labels may not be used for either).

    from src.external import metrica, dfl, degrade_ball
    d = metrica(1)  -> {"players": (F,N,3), "ball": (F,3), "team": (N,), "labels": (F,) in "HAD", "fps": 25}

Metrica sample games 1-2 (same provider/convention as BAR-ATH): labels derived from the event
stream -- team of the last on-ball event; dead from BALL OUT / foul / goal until the next SET PIECE.
DFL (floodlight IDSSE): labels straight from its per-frame possession + ballstatus codes.
Both are algorithmic derivations of other games' data: allowed, and documented in DATA_SOURCES.md.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
MET = os.path.join(ROOT, "data_raw/external/metrica-sample/data")
ON_BALL = {"PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST", "FAULT RECEIVED"}   # team of this event has the ball

def metrica(game=1, L=105.0, W=68.0):
    import pandas as pd
    g = f"Sample_Game_{game}"; tr = {}
    for side in ("Home", "Away"):
        t = pd.read_csv(f"{MET}/{g}/{g}_RawTrackingData_{side}_Team.csv", skiprows=2)
        cols = [c for c in t.columns if c.startswith("Player")]
        xy = np.stack([t[[c, t.columns[t.columns.get_loc(c) + 1]]].values for c in cols], 1).astype(np.float32)  # (F,N,2)
        tr[side] = xy; ball = t[["Ball", t.columns[-1]]].values.astype(np.float32); period = t["Period"].values
    xy = np.concatenate([tr["Home"], tr["Away"]], 1) * np.array([L, W], np.float32)
    F, N = xy.shape[:2]
    players = np.concatenate([xy, np.full((F, N, 1), np.nan, np.float32)], 2)
    team = np.array([0] * tr["Home"].shape[1] + [1] * tr["Away"].shape[1], np.int8)
    ballm = np.concatenate([ball * np.array([L, W], np.float32), np.full((F, 1), np.nan, np.float32)], 1)
    e = pd.read_csv(f"{MET}/{g}/{g}_RawEventsData.csv"); e = e.sort_values("Start Frame")
    lab = np.full(F, 2, np.int8)                        # 0=H 1=A 2=D ; frames are 1-based in Metrica
    cur = 2
    for _, r in e.iterrows():
        f0 = int(r["Start Frame"]) - 1
        if r["Type"] in ON_BALL: cur = 0 if r["Team"] == "Home" else 1
        elif r["Type"] == "CHALLENGE" and isinstance(r["Subtype"], str) and r["Subtype"].endswith("WON"): cur = 0 if r["Team"] == "Home" else 1
        lab[f0:] = cur
        if r["Type"] in ("BALL OUT", "FAULT RECEIVED") or (r["Type"] == "SHOT" and "GOAL" in str(r["Subtype"]) and "OFF" not in str(r["Subtype"])):
            lab[f0:] = 2
    for p in (1, 2):                                    # before each kickoff is dead
        i0 = np.argmax(period == p); lab[i0:i0 + int(np.argmax(lab[i0:] != 2))] = 2
    return {"players": players, "ball": ballm, "team": team, "labels": np.array(list("HAD"))[lab], "fps": 25, "period": period}

def dfl(match_id="J03WMX"):
    from floodlight.io.datasets import IDSSEDataset
    ev, xy, pos, bs, ts, pitch = IDSSEDataset(dataset_dir_name="idsse", match_id=match_id).get(match_id)
    P, B, LAB, PER = [], [], [], []
    for k, seg in enumerate(("firstHalf", "secondHalf")):
        h, a, b = xy[seg]["Home"].xy, xy[seg]["Away"].xy, xy[seg]["Ball"].xy
        F = len(b); hp = h.reshape(F, -1, 2); ap = a.reshape(F, -1, 2)
        p = np.concatenate([hp, ap], 1); P.append(np.concatenate([p, np.full((F, p.shape[1], 1), np.nan)], 2))
        B.append(np.concatenate([b[:, :2], np.full((F, 1), np.nan)], 1))
        code_p, code_b = pos[seg].code, bs[seg].code                       # possession 1=Home 2=Away ; ballstatus 1=alive 0=dead
        lab = np.where(code_b == 0, 2, np.where(code_p == 1, 0, 1)); LAB.append(lab); PER.append(np.full(F, k + 1))
        nh, na = hp.shape[1], ap.shape[1]
    players = np.concatenate(P).astype(np.float32); ball = np.concatenate(B).astype(np.float32)
    x0, y0 = pitch.xlim[0], pitch.ylim[0]; players[:, :, 0] -= x0; players[:, :, 1] -= y0; ball[:, 0] -= x0; ball[:, 1] -= y0   # origin to corner like the ATD
    return {"players": players, "ball": ball, "team": np.array([0] * nh + [1] * na, np.int8), "labels": np.array(list("HAD"))[np.concatenate(LAB)],
            "fps": xy["firstHalf"]["Ball"].framerate, "period": np.concatenate(PER), "pitch": (pitch.length, pitch.width)}

def dead_ball_carry(d, seed=0, sigma=2.0, step=0.05, p_carry=0.6):
    """DFL freezes the ball while dead (speed 0 on 100% of dead frames, ~5 m from everyone); the BAR-ATH tracker keeps
    reporting a moving ball during stoppages (median 2.7 m/s, 35% within 1.5 m of a player, p75 6.5 m away). Per dead run:
    with p_carry attach the ball to the nearest player at the stoppage (random-walk offset), else let it drift from where
    it stopped (random walk). Either way the model cannot learn 'still = dead' or 'glued to a walker = dead'."""
    r = np.random.default_rng(seed); out = dict(d); b = d["ball"].copy(); pl = d["players"][:, :, :2]; dead = d["labels"] == "D"
    edges = np.diff(np.r_[0, dead.astype(int), 0])
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        walk = r.normal(0, step, (e - s, 2)).cumsum(0)
        dist = np.linalg.norm(pl[s] - b[s, :2], axis=1); dist = np.where(np.isfinite(dist), dist, np.inf)
        if r.random() < p_carry and np.isfinite(dist).any(): b[s:e, :2] = pl[s:e, int(dist.argmin())] + r.normal(0, sigma, 2) + walk
        else: b[s:e, :2] = b[s, :2] + walk
    out["ball"] = b; return out

def simulate_detector(d, seed=0, fill=0.18, run_mean=7, p_gross=0.11, sigma=0.65):
    """Our YOLO ball detector (src/ball_detect.py) run through ball_fuse (conf 0.5, support 10/10) fills ~18% of the ATD's
    missing graded frames in runs of median 3 / p90 20 frames; accepted detections are 0.77 m off at the median and >5 m
    off 11% of the time, in runs of ~3 frames (eda/NOTES.md 2026-10-27). Reproduce that on a degraded DFL ball so fused-ball
    pipelines can be judged on DFL leave-one-out instead of the public window. Call after degrade_ball; d must still carry
    the true ball as d["ball_true"]. Other settings model a better detector (plans/soccernet_ball_detector.md step 1).
    Fills land on dead frames too: sparing them leaks the label (a still-missing ball then means dead)."""
    r = np.random.default_rng(seed + 7); out = dict(d); b = d["ball"].copy(); t = d["ball_true"]; hidden = ~np.isfinite(b[:, 0]) & np.isfinite(t[:, 0]); F = len(b)
    starts = hidden & (r.random(F) < fill / run_mean)
    for s in np.where(starts)[0]:
        e = min(F, s + r.geometric(1 / run_mean)); seg = slice(s, e); m = hidden[seg]
        err = r.normal(0, sigma, 2) if r.random() >= p_gross else r.normal(0, 1, 2) / np.sqrt(2) * r.uniform(5, 20)
        idx = np.where(m)[0] + s; b[idx, :2] = t[idx, :2] + err; hidden[seg] = False
    out["ball"] = b; return out

# BAR-ATH tactical ATD, graded frames (eda/NOTES.md "Make DFL look like BAR-ATH", 2026-10-28): run-length percentiles in frames
Q = np.array([0, 5, 10, 25, 50, 75, 90, 95, 99, 100])
BALL_SEEN = np.array([1, 40, 53, 135, 343, 762, 1621, 2292, 3222, 4000])     # visible runs, mean 633 (59% visible)
BALL_GAP = np.array([1, 2, 67, 154, 268, 528, 882, 1273, 2687, 3500])        # gaps, mean 439: either < 1 s or > 3 s, never 1-3 s
PLAYER_BAD = np.array([1, 8, 15, 33, 73, 167, 682, 1200, 2500, 4000])       # stretches with >= 1 of 22 players untracked, mean 259; 48% of frames
PLAYER_GOOD_MEAN = 175                                                       # tuned so 48% of frames miss someone (52% have all 22)
N_MISSING = {1: 0.54, 2: 0.16, 3: 0.07, 4: 0.04, 5: 0.07, 6: 0.06, 8: 0.06}  # players missing inside a bad stretch (5+ bucket spread)

def draw(r, table, n):
    """n run lengths from a percentile table (inverse-CDF interpolation)."""
    return np.interp(r.random(n) * 100, Q, table).astype(int).clip(1)

def degrade_ball(d, visible=None, seed=0):
    """Hide the DFL ball with BAR-ATH's real gap structure (run lengths drawn from the ATD's percentile tables, so gaps
    are either sub-second or 3 s+ and visibility lands at ~59%). `visible` is ignored (kept for the CLI)."""
    r = np.random.default_rng(seed); F = len(d["ball"]); out = dict(d); b = d["ball"].copy(); i = r.integers(0, 300)
    while i < F:
        on, off = draw(r, BALL_SEEN, 1)[0], draw(r, BALL_GAP, 1)[0]; b[i + on:i + on + off] = np.nan; i += on + off
    out["ball"] = b; out["ball_true"] = d["ball"]; return out

def degrade_ball_geom(d, visible=0.57, seed=0):
    """The pre-2026-10-28 degradation (geometric gaps, mean 1 s): kept for the train-old/test-real comparison in eda/dead_diag.py."""
    r = np.random.default_rng(seed); F = len(d["ball"]); out = dict(d); b = d["ball"].copy(); i = 0
    while i < F:
        on = r.geometric(1 / (25 * visible / (1 - visible))); off = r.geometric(1 / 25); b[i + on:i + on + off] = np.nan; i += on + off
    out["ball"] = b; out["ball_true"] = d["ball"]; return out

def degrade_players(d, seed=0):
    """Untrack players the way the BAR-ATH tracklets do: alternating good stretches (all 22) and bad stretches (1-8 players
    missing for the whole stretch). Identity swaps are not simulated: slots never change team in the ATD and our features
    are slot-agnostic, so only presence matters. ponytail: missing players are chosen uniformly; real tracklets end nearer the
    touchline (median 12 m vs 18 m) -- add an edge preference if PathCRF training ever uses this."""
    r = np.random.default_rng(seed + 11); out = dict(d); P = d["players"].copy(); F, N = P.shape[:2]; i = r.integers(0, 300)
    ks, ps = np.array(list(N_MISSING)), np.array(list(N_MISSING.values())); ps /= ps.sum()
    while i < F:
        bad = draw(r, PLAYER_BAD, 1)[0]; m = r.choice(ks, p=ps); P[i:i + bad, r.choice(N, min(m, N), replace=False)] = np.nan
        i += bad + r.geometric(1 / PLAYER_GOOD_MEAN)
    out["players"] = P; return out

if __name__ == "__main__":
    d = metrica(1); L = d["labels"]
    print("metrica1", d["players"].shape, "ball visible %.2f" % np.isfinite(d["ball"][:, 0]).mean(), {k: round((L == k).mean(), 3) for k in "HAD"})
    sw = (L[1:] != L[:-1]).sum(); print("label switches", sw, "mean spell %.1fs" % (len(L) / sw / 25))
    assert d["players"].shape[1] == len(d["team"]) and len(L) == len(d["ball"])
    print("degraded visible %.2f" % np.isfinite(degrade_ball(d)["ball"][:, 0]).mean())
