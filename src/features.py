"""Per-frame features for the v2 classifier. Slot-agnostic (per-team nearest distances, counts,
spread), so DFL (22 clean tracks, ball always seen) and BAR-ATH (46 tracklet slots, ball 57%) look
alike once the DFL ball is degraded. Everything is in metres on a corner-origin pitch.

    X, names = features(d, pitch)      # (F, n_features) float32
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.possession import fill_ball

def roll(x, half):
    """Centred moving mean over 2*half+1 frames, NaN-aware."""
    v = np.where(np.isfinite(x), x, 0.0); c = np.isfinite(x).astype(float)
    k = np.ones(2 * half + 1); s = np.convolve(v, k, "same"); n = np.convolve(c, k, "same")
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)

SPACE = os.environ.get("SPACE", "0") == "1"   # team-shape features (convex hull, Voronoi share); experiment, eda/NOTES.md

def space(pl, team, b, pitch, n=(21, 14), r=15.0):
    """Per team: convex hull area (m^2, NaN under 3 players), share of the pitch it controls (Voronoi = nearest player, on a
    grid), and its share of grid points within r m of the ball. One frame of 22 players costs ~0.1 ms."""
    from scipy.spatial import ConvexHull
    L, W = pitch; gx, gy = np.meshgrid((np.arange(n[0]) + 0.5) * L / n[0], (np.arange(n[1]) + 0.5) * W / n[1]); G = np.c_[gx.ravel(), gy.ravel()]
    F = len(pl); hull = np.full((F, 2), np.nan); vor = np.full((F, 2), np.nan); near = np.full((F, 2), np.nan)
    for i in range(0, F, 2000):
        P = pl[i:i + 2000]; d = np.linalg.norm(G[None, :, None] - P[:, None], axis=3); d = np.where(np.isfinite(d), d, np.inf)
        own = np.take_along_axis(np.broadcast_to(team, P.shape[:2]), d.argmin(2), 1)            # owning team per grid point
        own = np.where(np.isfinite(d.min(2)), own, -1); nb = np.linalg.norm(G[None] - b[i:i + 2000, None], axis=2) < r
        for t in (0, 1):
            vor[i:i + 2000, t] = (own == t).mean(1); near[i:i + 2000, t] = np.where(nb.any(1), ((own == t) & nb).sum(1) / nb.sum(1).clip(1), np.nan)
    for f in range(F):
        for t in (0, 1):
            q = pl[f][(team == t) & np.isfinite(pl[f, :, 0])]
            if len(q) >= 3:
                try: hull[f, t] = ConvexHull(q).volume
                except Exception: pass                                                          # collinear
    return hull, vor, near

def features(d, pitch, gap=50):
    pl, team = d["players"][:, :, :2], d["team"]; L, W = pitch; fps = d.get("fps", 25)
    raw = d["ball"][:, :2]; seen = np.isfinite(raw[:, 0])
    b = fill_ball(raw, gap); known = np.isfinite(b[:, 0]); F = len(b)
    # frames since the ball was last seen / until next seen (capped 10 s)
    idx = np.arange(F); last = np.maximum.accumulate(np.where(seen, idx, -1)); since = np.where(last >= 0, idx - last, 250).clip(0, 250)
    nxt = np.minimum.accumulate(np.where(seen, idx, F)[::-1])[::-1]; until = np.where(nxt < F, nxt - idx, 250).clip(0, 250)
    # last known ball position carried forward, for cluster features during gaps
    bc = b.copy()
    for k in range(2): bc[:, k] = np.where(known, b[:, k], np.nan); bc[:, k] = np.interp(idx, idx[known], b[known, k]) if known.any() else bc[:, k]
    dist = np.linalg.norm(pl - bc[:, None, :], axis=2)
    out = {"known": known.astype(float), "seen": seen.astype(float), "since": since / 25.0, "until": until / 25.0}
    for nm, t in (("H", 0), ("A", 1)):
        dt = np.where(team == t, dist, np.nan); srt = np.sort(np.where(np.isfinite(dt), dt, 99.0), axis=1)
        out[f"d1{nm}"] = srt[:, 0].clip(0, 30); out[f"d2{nm}"] = srt[:, 1].clip(0, 30)
        out[f"n3{nm}"] = (dt < 3).sum(1).astype(float); out[f"n8{nm}"] = (dt < 8).sum(1).astype(float)
    out["ddiff"] = out["d1H"] - out["d1A"]
    speed = np.r_[0, np.linalg.norm(np.diff(b, axis=0), axis=1)] * fps; out["speed"] = np.where(known, speed, np.nan).clip(0, 40)
    out["bx"] = bc[:, 0] / L; out["by"] = bc[:, 1] / W
    out["off"] = (known & ((b[:, 0] < -0.5) | (b[:, 0] > L + 0.5) | (b[:, 1] < -0.5) | (b[:, 1] > W + 0.5))).astype(float)
    out["edge"] = np.minimum(np.minimum(bc[:, 0], L - bc[:, 0]), np.minimum(bc[:, 1], W - bc[:, 1])).clip(-5, 20)
    ok = np.isfinite(pl[:, :, 0]); n = ok.sum(1).clip(1); cen = np.nansum(np.where(ok[:, :, None], pl, 0), 1) / n[:, None]
    out["ntracked"] = n.astype(float); out["spread"] = np.sqrt(np.nansum(np.where(ok, ((pl - cen[:, None]) ** 2).sum(2), 0), 1) / n).clip(0, 50)
    out["dcen"] = np.linalg.norm(bc - cen, axis=1).clip(0, 60)
    out["pspeed"] = np.r_[0, np.nanmean(np.linalg.norm(np.diff(pl, axis=0), axis=2), 1)] * fps   # mean player speed (walking = dead)
    def since(ev):                                                            # seconds since the event last happened (capped 10 s), and until it next happens
        last = np.maximum.accumulate(np.where(ev, idx, -1)); nx = np.minimum.accumulate(np.where(ev, idx, F)[::-1])[::-1]
        return np.where(last >= 0, idx - last, 250).clip(0, 250) / 25.0, np.where(nx < F, nx - idx, 250).clip(0, 250) / 25.0
    for nm in ("H", "A"): out[f"sinceT{nm}"], out[f"untilT{nm}"] = since(known & (out[f"d1{nm}"] < 1.5))     # touch = that team within 1.5 m of a seen ball
    out["sinceOff"], out["untilOff"] = since(out["off"] > 0)
    kick = known & (np.abs(np.diff(np.r_[speed[0], speed])) > 4); out["sinceKick"], out["untilKick"] = since(kick)   # speed jump > 4 m/s per frame
    if SPACE:
        hull, vor, near = space(pl, team, bc, pitch)
        for t, nm in enumerate("HA"): out[f"hull{nm}"] = hull[:, t] / 100; out[f"vor{nm}"] = vor[:, t]; out[f"near{nm}"] = near[:, t]
    base = dict(out)
    for half in (12, 50):                                                     # ±0.5 s and ±2 s context (±5 s tried: no gain on DFL CV)
        for k in ("d1H", "d1A", "ddiff", "speed", "known", "pspeed", "n3H", "n3A"): out[f"{k}_r{half}"] = roll(base[k], half)
        if SPACE:
            for k in ("hullH", "hullA", "vorH", "vorA", "nearH", "nearA"): out[f"{k}_r{half}"] = roll(base[k], half)
    names = list(out); X = np.column_stack([out[k] for k in names]).astype(np.float32)
    return X, names

if __name__ == "__main__":
    from src.external import metrica, degrade_ball
    d = degrade_ball(metrica(1)); X, names = features(d, (105.0, 68.0))
    assert X.shape == (len(d["ball"]), len(names)) and np.isfinite(X[:, names.index("d1H")]).all()
    print(X.shape, names); print("nan fraction per feature:", {k: round(float(np.isnan(X[:, i]).mean()), 2) for i, k in enumerate(names) if np.isnan(X[:, i]).any()})
