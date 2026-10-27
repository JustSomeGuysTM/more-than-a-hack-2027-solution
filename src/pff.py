"""PFF FC World Cup 2022 (public, 64 matches, tracking 29.97 fps + event data) -> the same dict as
external.dfl() at 25 fps (npz cache) and PathCRF 'tracking_processed' parquet.

    .venv/bin/python src/pff.py --npz [--jobs 8]      # data_raw/external/pff_<gid>.npz for every game
    .venv/bin/python src/pff.py --pathcrf [--jobs 8]  # ~/pathcrf/data/pff/tracking_processed/<gid>.parquet
    from src.pff import pff; d = pff("10502")

Labels are algorithmic (no human label on our match is involved): D from each period start until
the first on-ball event, from an OUT event, a foul or a goal until the next on-ball event, and after
END; otherwise the team of the last on-ball (OTB) event. On 10502/10503 that gives 35% dead in
93-125 dead spells of 16-23 s (DFL ground truth: 42%, 92-103 spells of 23-27 s).
"""
import argparse, bz2, glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
PFF = os.path.join(ROOT, "data_raw/external/pff/FIFA World Cup 2022"); CACHE = os.path.join(ROOT, "data_raw/external")
PATHCRF = os.path.expanduser("~/pathcrf/data/pff/tracking_processed"); FPS_IN, FPS = 29.97, 25; PITCH = (105.0, 68.0)

def games(): return sorted(os.path.basename(f).split(".")[0] for f in glob.glob(f"{PFF}/Tracking Data/*.jsonl.bz2"))

def raw(gid):
    """Parse one tracking file: per native frame -> period, frameNum, per-team {jersey: xy}, ball xyz, OTB (home?, jersey) or event type."""
    per, fn, home, away, ball, ev = [], [], [], [], [], []
    with bz2.open(f"{PFF}/Tracking Data/{gid}.jsonl.bz2", "rt") as f:
        for line in f:
            d = json.loads(line); g = d.get("game_event"); b = (d.get("balls") or [{}])[0]
            per.append(d["period"]); fn.append(d["frameNum"])
            home.append({p["jerseyNum"]: (p["x"], p["y"]) for p in d["homePlayers"] or []}); away.append({p["jerseyNum"]: (p["x"], p["y"]) for p in d["awayPlayers"] or []})
            ball.append((b.get("x"), b.get("y"), b.get("z")) if b.get("x") is not None else (np.nan,) * 3)
            ev.append((g["game_event_type"], int(g.get("home_team") or 0), str(g.get("shirt_number"))) if g else None)
    return np.array(per), np.array(fn), home, away, np.array(ball, np.float32), ev

def labels(per, fn, ev, gid):
    """0=H 1=A 2=D per native frame, by the rule in the module docstring."""
    n = len(per); typ = np.array([e[0] if e else None for e in ev], object); otb = np.flatnonzero(typ == "OTB"); dead = np.zeros(n, bool)
    def dead_until_otb(i):
        nxt = otb[otb > i]; dead[max(i, 0):(nxt[0] if len(nxt) else n)] = True
    for p in np.unique(per): dead_until_otb(int(np.argmax(per == p)) - 1)
    for i in np.flatnonzero(typ == "OUT"): dead_until_otb(i)
    for x in json.load(open(f"{PFF}/Event Data/{gid}.json")):
        g, pe, fl = x["gameEvents"], x.get("possessionEvents") or {}, x.get("fouls") or {}
        if any(v for v in fl.values()) or g.get("endType") == "G" or pe.get("shotOutcomeType") == "G":
            dead_until_otb(int(np.searchsorted(fn, round((x.get("endTime") or x["startTime"]) * FPS_IN))))   # frameNum == video time * fps
    e = np.flatnonzero(typ == "END")
    if len(e): dead[e[0]:] = True
    team = np.full(n, -1); team[otb] = [1 - ev[i][1] for i in otb]            # home_team flag 1 -> H (0)
    for i in range(1, n):
        if team[i] < 0: team[i] = team[i - 1]
    return np.where(dead, 2, np.clip(team, 0, 1))

def resample(per, fn):
    """Native 29.97 fps -> 25 fps: nearest native frame to each 1/25 s tick, per period."""
    idx = []
    for p in np.unique(per):
        i = np.flatnonzero(per == p); t = (fn[i] - fn[i[0]]) / FPS_IN; ticks = np.arange(0, t[-1], 1 / FPS)
        idx.append(i[np.searchsorted(t, ticks)])
    return np.concatenate(idx)

def pff(gid):
    cache = f"{CACHE}/pff_{gid}.npz"
    if os.path.exists(cache): z = np.load(cache); d = {k: z[k] for k in z.files}; d["fps"] = FPS; d["pitch"] = PITCH; return d
    per, fn, home, away, ball, ev = raw(gid); lab = labels(per, fn, ev, gid); idx = resample(per, fn)
    hj = sorted({j for h in home for j in h}, key=int); aj = sorted({j for a in away for j in a}, key=int); slots = [("h", j) for j in hj] + [("a", j) for j in aj]
    P = np.full((len(idx), len(slots), 3), np.nan, np.float32)
    for r, i in enumerate(idx):
        for s, (side, j) in enumerate(slots):
            xy = (home if side == "h" else away)[i].get(j)
            if xy: P[r, s, :2] = xy
    P[:, :, 0] += PITCH[0] / 2; P[:, :, 1] += PITCH[1] / 2; B = ball[idx].copy(); B[:, 0] += PITCH[0] / 2; B[:, 1] += PITCH[1] / 2   # corner origin like the ATD / DFL
    otb_player = np.array([f"{'home' if e[1] else 'away'}_{e[2]}" if e and e[0] == "OTB" else "" for e in ev])
    d = {"players": P, "ball": B, "team": np.array([0] * len(hj) + [1] * len(aj), np.int8), "labels": np.array(list("HAD"))[lab[idx]], "period": per[idx],
         "slots": np.array([f"{'home' if s == 'h' else 'away'}_{j}" for s, j in slots]), "otb_player": otb_player[idx]}
    np.savez_compressed(cache, **d); d["fps"] = FPS; d["pitch"] = PITCH; return d

def pathcrf(gid):
    """25-fps npz -> PathCRF tracking_processed parquet (phase = episode = alive run >= 10 s, like src/pathcrf_io.py)."""
    import pandas as pd; from src.possession import fill_ball
    sys.path.insert(0, os.path.expanduser("~/pathcrf")); from datatools.utils import calculate_running_features
    d = pff(gid); F = len(d["ball"]); lab = d["labels"]; alive = lab != "D"; per = d["period"]
    ball = pd.DataFrame(fill_ball(d["ball"][:, :2], 10 ** 9)).ffill().bfill().values
    ts = np.zeros(F)
    for p in np.unique(per): i = np.flatnonzero(per == p); ts[i] = (i - i[0]) / FPS
    own = pd.Series(np.where(lab == "H", "home", np.where(lab == "A", "away", None))).ffill().bfill().values
    df = pd.DataFrame({"frame_id": np.arange(F), "period_id": per, "timestamp": ts, "ball_state": np.where(alive, "alive", "dead"), "ball_owning_team_id": own,
                       "player_id": np.where(alive & (d["otb_player"] != ""), d["otb_player"], None), "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_z": np.nan_to_num(d["ball"][:, 2])})
    ep = np.zeros(F, int); starts = np.flatnonzero(np.diff(np.r_[0, alive.astype(int)]) == 1)
    for k, s in enumerate(starts, 1):
        e = s + int(np.argmax(~alive[s:])) if (~alive[s:]).any() else F
        if e - s >= 10 * FPS: ep[s:e] = k
    df["episode_id"] = ep; df["phase_id"] = ep
    for s, name in enumerate(d["slots"]): df[f"{name}_x"] = d["players"][:, s, 0]; df[f"{name}_y"] = d["players"][:, s, 1]
    df = calculate_running_features(df); os.makedirs(PATHCRF, exist_ok=True); df.to_parquet(f"{PATHCRF}/{gid}.parquet"); return gid, df.shape, int(ep.max())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", action="store_true"); ap.add_argument("--pathcrf", action="store_true"); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--games", nargs="*")
    a = ap.parse_args(); gs = a.games or games(); from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        if a.npz:
            for d, g in zip(pool.imap_unordered(pff, gs), gs): pass
            print("npz done", len(gs))
        if a.pathcrf:
            for r in pool.imap_unordered(pathcrf, gs): print(r, flush=True)
