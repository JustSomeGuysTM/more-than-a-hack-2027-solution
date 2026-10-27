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

def degrade_ball(d, visible=0.57, seed=0):
    """Hide the ball with BAR-ATH-like gaps (mean gap ~1 s, geometric) so a model trained here does
    not learn to trust a ball it will not see. ponytail: gap stats are a guess; fit to the ATD later."""
    r = np.random.default_rng(seed); F = len(d["ball"]); out = dict(d); b = d["ball"].copy(); i = 0
    while i < F:
        on = r.geometric(1 / (25 * visible / (1 - visible))); off = r.geometric(1 / 25)
        b[i + on:i + on + off] = np.nan; i += on + off
    out["ball"] = b; return out

if __name__ == "__main__":
    d = metrica(1); L = d["labels"]
    print("metrica1", d["players"].shape, "ball visible %.2f" % np.isfinite(d["ball"][:, 0]).mean(), {k: round((L == k).mean(), 3) for k in "HAD"})
    sw = (L[1:] != L[:-1]).sum(); print("label switches", sw, "mean spell %.1fs" % (len(L) / sw / 25))
    assert d["players"].shape[1] == len(d["team"]) and len(L) == len(d["ball"])
    print("degraded visible %.2f" % np.isfinite(degrade_ball(d)["ball"][:, 0]).mean())
