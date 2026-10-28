"""BAR-ATH tactical ATD -> PathCRF 'tracking_processed' parquet.

    .venv/bin/python src/pathcrf_io.py [--alive evaluation/results/v2.csv]   # -> data_raw/pathcrf/BARATH.parquet

PathCRF wants, per alive episode, a NaN-free set of <= 11 identified players per team. Our ATD has 46
tracklet slots with ID churn, so: (1) stitch slots into persistent tracks per team (a slot segment that
starts within 5 s and 8 m/s of a track that just ended continues it), (2) interpolate gaps <= 2 s,
(3) inside each episode hold the last known position for longer gaps, (4) phase = episode so the
active-player set is per episode. Alive/dead comes from our own model's CSV (algorithmic, allowed).
"""
import argparse, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from eda.atd import load
from src.possession import fill_ball
OUT = os.path.join(ROOT, "data_raw/pathcrf"); FPS = 25; MAX_GAP_S, MAX_SPEED, N_TRACK = 5.0, 8.0, 11
HALVES = {1: (11185, 90652), 2: (113742, 190198)}

def stitch(P, slots):
    """P (F,46,2); slots of one team -> (F, N_TRACK, 2) persistent tracks, NaN where nobody.
    Segments of all slots are assigned in start order, so a track's `last_f` is truly its last end."""
    F = len(P); T = np.full((F, N_TRACK, 2), np.nan, np.float32); last_f = np.full(N_TRACK, -10 ** 9); last_xy = np.zeros((N_TRACK, 2)); segs = []
    for s in slots:
        ok = np.isfinite(P[:, s, 0]); edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0])); segs += [(a, b, s) for a, b in zip(edges[::2], edges[1::2])]
    dropped = 0
    for a, b, s in sorted(segs):
        gap = (a - last_f) / FPS; dist = np.linalg.norm(last_xy - P[a, s, :2], axis=1)
        cand = np.where((gap > 0) & (gap <= MAX_GAP_S) & (dist <= MAX_SPEED * np.maximum(gap, 0.2)), dist, np.inf)
        if np.isfinite(cand).any(): k = int(cand.argmin())
        else:
            free = np.flatnonzero(last_f < a)
            if not len(free): dropped += b - a; continue            # > 11 simultaneous tracklets (bench, duplicates): drop the extra
            k = int(free[np.argmin(last_f[free])])
        T[a:b, k] = P[a:b, s, :2]; last_f[k] = b - 1; last_xy[k] = P[b - 1, s, :2]
    print(f"stitch: {len(segs)} segments, dropped {dropped / 25:.0f} s of tracklets beyond 11 per team")
    return T

def interp_gaps(T, max_gap):
    for k in range(T.shape[1]): T[:, k] = fill_ball(T[:, k], max_gap)
    return T

def main(alive_csv, source="tactical"):
    d = load(source); P, team, B = d["players"][:, :, :2], d["team"], d["ball"][:, :2]; F = len(P)
    lab = pd.read_csv(alive_csv)["label"].values; alive = np.zeros(F, bool)
    for h, (a, b) in HALVES.items(): alive[a:b] = lab[a:b] != "D"
    tracks = {"home": interp_gaps(stitch(P, np.flatnonzero(team == 0)), 50), "away": interp_gaps(stitch(P, np.flatnonzero(team == 1)), 50)}
    ball = fill_ball(B, 10 ** 9); ball = pd.DataFrame(ball).ffill().bfill().values
    rows = {"frame_id": np.arange(F), "period_id": 0, "timestamp": 0.0, "ball_state": np.where(alive, "alive", "dead"), "ball_owning_team_id": None, "player_id": None, "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_z": 0.0}
    df = pd.DataFrame(rows)
    for h, (a, b) in HALVES.items(): df.loc[a:b - 1, "period_id"] = h; df.loc[a:b - 1, "timestamp"] = (np.arange(a, b) - a) / FPS
    df = df[df.period_id > 0].reset_index(drop=True)
    ep = np.zeros(len(df), int); al = (df.ball_state == "alive").values; starts = np.flatnonzero(np.diff(np.r_[0, al.astype(int)]) == 1)
    for i, s in enumerate(starts, 1):
        e = s + int(np.argmax(~al[s:])) if (~al[s:]).any() else len(df); ep[s:e] = i
    df["episode_id"] = ep; df["phase_id"] = ep
    for side, T in tracks.items():
        T = T[df.frame_id.values]
        for k in range(N_TRACK):
            x = pd.DataFrame(T[:, k], columns=["x", "y"]); x["ep"] = ep
            x[["x", "y"]] = x.groupby("ep")[["x", "y"]].transform(lambda c: c.ffill().bfill())   # hold position inside an episode
            df[f"{side}_{k}_x"] = x["x"].values; df[f"{side}_{k}_y"] = x["y"].values
    # nearest-player guess as player_id so PathCRF's bookkeeping has something to read (not used as truth)
    cols = [c for c in df.columns if c.endswith("_x") and c[:4] in ("home", "away")]
    xy = np.stack([df[[c, c[:-2] + "_y"]].values for c in cols], 1); dist = np.linalg.norm(xy - df[["ball_x", "ball_y"]].values[:, None], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf); near = dist.argmin(1); df["player_id"] = np.where(np.isfinite(dist).any(1), np.array([c[:-2] for c in cols])[near], None)
    sys.path.insert(0, os.path.expanduser("~/pathcrf")); from datatools.utils import calculate_running_features
    df = calculate_running_features(df)
    # episodes shorter than PathCRF's 10 s window are skipped by it anyway; zero them so phase bookkeeping never sees an empty slice
    short = df.groupby("episode_id")["frame_id"].transform("size") < 10 * FPS; df.loc[short | (df.episode_id == 0), ["episode_id", "phase_id"]] = 0
    os.makedirs(OUT, exist_ok=True); df.to_parquet(f"{OUT}/BARATH.parquet"); print("wrote", f"{OUT}/BARATH.parquet", df.shape)
    n_ep = df.episode_id.max(); lens = df[df.episode_id > 0].groupby("episode_id").size() / FPS
    print(f"episodes {n_ep}, >=10 s: {(lens >= 10).sum()} covering {lens[lens >= 10].sum() / lens.sum():.0%} of alive time")
    for side in ("home", "away"):
        n = np.isfinite(df[[f"{side}_{k}_x" for k in range(N_TRACK)]].values).sum(1); print(f"{side}: players per alive frame mean {n[al].mean():.1f}, p10 {np.percentile(n[al], 10):.0f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--alive", default=os.path.join(ROOT, "evaluation/results/seq.csv")); ap.add_argument("--source", default="tactical"); a = ap.parse_args(); main(a.alive, a.source)
