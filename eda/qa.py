"""QA for the possession pipeline (plans/qa_plots.md).

Out-of-fold predictions on the 7 DFL matches, where the official labels are the truth: typed errors, oracle fixes that
say where spellF1 is lost, per-match timelines and clips of the worst errors. Plus a PFF rule-vs-teacher label check
and a spell-change index video for BAR-ATH.

Watching is fine; nothing here may turn into labels, corrections or frame-specific rules on BAR-ATH (RULES.md, rule 1).
Everything is written to eda/out/qa/ (gitignored) and BAR-ATH renders never leave the laptop.

    # PathCRF fold predictions first (PathCRF venv; fold k = trial 200+k holds out IDS[k]):
    for t in 200:J03WMX 201:J03WN1 202:J03WOH 203:J03WOY 204:J03WPY 205:J03WQQ 206:J03WR9; do
        ~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial ${t%:*} --dfl ${t#*:} --save; done
    # BiGRU out-of-fold dead balls come from eda/out/seq_folds.npz (src/seq.py --cv)
    .venv/bin/python eda/qa.py dfl                                    # table, eda/out/qa/errors.csv, timeline_<match>.png
    .venv/bin/python eda/qa.py clips --type dead_onset_late --n 5     # 2D clips of the worst errors of one type
    .venv/bin/python eda/qa.py pff 10502 [--clips 5]                  # PFF rule labels vs teacher labels
    .venv/bin/python eda/qa.py barath --csv evaluation/results/pathcrf.csv [--video | --checks-only]   # sanity checks + spell-change video
    python3 eda/qa.py selftest                                        # synthetic checks, no data needed
"""
import argparse, csv, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "eda/out/qa"); EXT = os.path.join(ROOT, "data_raw/external")
SEQ_FOLDS = os.path.join(ROOT, "eda/out/seq_folds.npz")
IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]   # src/v2.IDS order = seq_folds order = PathCRF fold k (trial 200+k)
GRADED = ((11185, 90652), (110808, 190198))                                      # BAR-ATH graded frames, [start, stop) per half
FPS = 25
TOL = 2 * FPS        # a GT possession change counts as found if the prediction switches to that team within +-2 s
LAG = 1 * FPS        # dead-ball onsets / restarts more than 1 s off are reported as late / early
MIN_SPELL = 2 * FPS  # wrong-team check only on GT possessions of at least 2 s
LAB = np.array(list("HAD"))
COL = {"H": "#d62728", "A": "#1f77b4", "D": "#9e9e9e", "V": "#2e7d32"}           # anything else (X, N, '') is white
TYPES = {"missed_stoppage": "#000000", "spurious_stoppage": "#7b1fa2", "dead_onset_late": "#ef6c00", "dead_onset_early": "#ffb74d",
         "restart_late": "#00838f", "restart_early": "#80deea", "missed_change": "#c2185b", "spurious_change": "#f48fb1",
         "wrong_team_spell": "#fdd835"}


# ---------------------------------------------------------------------------------------------- error typing (numpy only)

def runs(lab):
    """[(start, stop, label)] maximal constant runs."""
    lab = np.asarray(lab)
    if len(lab) == 0: return []
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return [(int(s), int(e), str(lab[s])) for s, e in zip(np.r_[0, cut], np.r_[cut, len(lab)])]

def switches_into(lab, team, a, b):
    """True if lab switches into `team` at some frame u with a <= u < b (lab[u-1] != team == lab[u])."""
    a, b = max(a, 1), min(b, len(lab))
    if b <= a: return False
    seg = lab[a - 1:b]
    return bool(np.any((seg[1:] == team) & (seg[:-1] != team)))

def run_length_from(lab, u):
    """Frames from u until lab changes."""
    nxt = np.flatnonzero(lab[u:] != lab[u]); return int(nxt[0]) if nxt.size else len(lab) - u

def classify(g, p, match=""):
    """Typed disagreements between ground truth g and prediction p (arrays of H/A/D). One dict per event; `seconds` is
    the size used to rank events (duration of the error, or how long the wrong label lasted)."""
    g, p = np.asarray(g).astype(str), np.asarray(p).astype(str); ev = []
    def add(t, s, e, size, detail=""):
        ev.append(dict(match=match, type=t, start=int(s), stop=int(e), seconds=round(size / FPS, 2), detail=detail))
    gd = [(s, e) for s, e, l in runs(g) if l == "D"]; pdd = [(s, e) for s, e, l in runs(p) if l == "D"]
    for s, e in gd:                                                   # each real stoppage: found? onset / restart timing
        ov = [(a, b) for a, b in pdd if a < e and b > s]
        if not ov: add("missed_stoppage", s, e, e - s, f"{(e - s) / FPS:.1f}s stoppage"); continue
        on, off = ov[0][0] - s, ov[-1][1] - e
        if on > LAG: add("dead_onset_late", s, ov[0][0], on, f"{on / FPS:+.1f}s")
        elif on < -LAG: add("dead_onset_early", ov[0][0], s, -on, f"{on / FPS:+.1f}s")
        if off > LAG: add("restart_late", e, ov[-1][1], off, f"{off / FPS:+.1f}s")
        elif off < -LAG: add("restart_early", ov[-1][1], e, -off, f"{off / FPS:+.1f}s")
    for a, b in pdd:                                                  # stoppages we invented
        if not any(a < e and b > s for s, e in gd): add("spurious_stoppage", a, b, b - a)
    ha = ["H", "A"]
    for u in np.flatnonzero((g[1:] != g[:-1]) & np.isin(g[1:], ha) & np.isin(g[:-1], ha)) + 1:   # live changes of possession
        if not switches_into(p, g[u], u - TOL, u + TOL + 1):
            spell = run_length_from(g, u); hit = np.flatnonzero(p[u:u + spell] == g[u]); lag = int(hit[0]) if hit.size else spell
            add("missed_change", u, u + lag, lag, f"{g[u - 1]}->{g[u]}, ours {'never' if not hit.size else f'{lag / FPS:.1f}s late'}")
    for u in np.flatnonzero((p[1:] != p[:-1]) & np.isin(p[1:], ha) & np.isin(p[:-1], ha)) + 1:   # switching back to the right team is not a new error
        if p[u] != g[u] and not switches_into(g, p[u], u - TOL, u + TOL + 1): add("spurious_change", u, u + run_length_from(p, u), run_length_from(p, u), f"{p[u - 1]}->{p[u]}")
    for s, e, l in runs(g):                                           # whole possessions given to the wrong team
        if l in ha and e - s >= MIN_SPELL:
            seg = p[s:e]; alive = np.isin(seg, ha)
            if alive.any() and (seg[alive] != l).mean() > 0.5: add("wrong_team_spell", s, e, e - s, f"gt {l}")
    return ev

def ffill_alive(p):
    """Each frame's most recent H/A label ('' before the first)."""
    idx = np.where(np.isin(p, ["H", "A"]), np.arange(len(p)), -1); idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, p[idx.clip(0)], "")

def oracle(g, p, what):
    """p with one error source fixed from g, to see what it is worth:
    onset = every found stoppage starts on time; dead = dead balls exactly right (frames we wrongly called dead keep our
    last team); team = the right team on every frame both call alive; both = dead + team (equals g)."""
    g, p = np.asarray(g).astype(str), np.asarray(p).astype(str); q = p.copy()
    if what == "onset":
        pdd = [(a, b) for a, b, l in runs(p) if l == "D"]
        for s, e, l in runs(g):
            ov = [a for a, b in pdd if l == "D" and a < e and b > s]
            if ov and ov[0] > s: q[s:ov[0]] = "D"
    if what in ("dead", "both"):
        q[g == "D"] = "D"; m = (g != "D") & (p == "D"); f = ffill_alive(p); q[m] = np.where(f[m] != "", f[m], g[m])
    if what in ("team", "both"):
        m = np.isin(g, ["H", "A"]) & np.isin(q, ["H", "A"]); q[m] = g[m]
    return q


# ---------------------------------------------------------------------------------------------- drawing

def rgb(lab):
    lab = np.asarray(lab).astype(str); out = np.ones((len(lab), 3))
    from matplotlib.colors import to_rgb
    for k, c in COL.items(): out[lab == k] = to_rgb(c)
    return out

def timeline(title, rows, ev, out, panel_min=10):
    """Label strips for a whole match in 10-minute panels, typed errors as ticks above them."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(rows); cols = {k: rgb(v) for k, v in rows.items()}; F = len(rows[names[0]]); P = int(panel_min * 60 * FPS)
    n = int(np.ceil(F / P)); R = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(18, 0.32 * (R + 2) * n + 0.8), squeeze=False)
    for k, ax in enumerate(axes[:, 0]):
        a, b = k * P, min((k + 1) * P, F)
        ax.imshow(np.stack([cols[x][a:b] for x in names]), aspect="auto", interpolation="nearest", extent=[a / FPS / 60, b / FPS / 60, R - 0.5, -0.5])
        for e in ev:
            if a <= e["start"] < b: ax.plot(e["start"] / FPS / 60, -1.0, "v", color=TYPES[e["type"]], ms=6, mec="k", mew=0.3)
        ax.set_ylim(R - 0.5, -1.5); ax.set_xlim(k * P / FPS / 60, (k + 1) * P / FPS / 60)
        ax.set_yticks(range(R)); ax.set_yticklabels(names, fontsize=8); ax.tick_params(axis="x", labelsize=8)
    axes[0, 0].set_title(title + "   (H red, A blue, D grey; ticks = typed errors, see legend; x = minutes)", fontsize=10)
    from matplotlib.lines import Line2D
    used = sorted({e["type"] for e in ev})
    if used: fig.legend([Line2D([], [], marker="v", ls="", color=TYPES[t], mec="k") for t in used], used, loc="lower center", ncol=len(used), fontsize=8)
    fig.tight_layout(rect=(0, 0.03 if used else 0, 1, 1)); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=90); plt.close(fig)
    print("wrote", out)

class Writer:
    """Frames -> mp4 through ffmpeg, opened on the first frame."""
    def __init__(self, out, fps): self.out, self.fps, self.p = out, fps, None
    def write(self, img):
        if self.p is None:
            h, w = img.shape[:2]; self.w, self.h = w - w % 2, h - h % 2; os.makedirs(os.path.dirname(self.out), exist_ok=True)
            self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.w}x{self.h}", "-r", str(self.fps),
                                       "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", self.out], stdin=subprocess.PIPE)
        self.p.stdin.write(np.ascontiguousarray(img[:self.h, :self.w]).tobytes())
    def close(self):
        if self.p is not None: self.p.stdin.close(); self.p.wait(); print("wrote", self.out)

class Pitch:
    """Pitch view of a cache-format match (players (F,N,3), ball (F,3), team (N,)) with label strips under it: +-10 s
    around the current frame, cursor in the middle, current labels in the corner. Optional camera frame on the left."""
    def __init__(self, pitch, strips):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from starter.utils import draw_pitch, HOME_COLOUR, AWAY_COLOUR
        self.plt, self.names, self.strips = plt, list(strips), {k: np.asarray(v).astype(str) for k, v in strips.items()}
        self.cols = {k: rgb(v) for k, v in self.strips.items()}; self.half = 10 * FPS
        self.fig = plt.figure(figsize=(9.6, 7.6), dpi=100); self.ax = self.fig.add_axes([0.03, 0.25, 0.94, 0.69]); draw_pitch(self.ax, *pitch)
        self.sax = self.fig.add_axes([0.1, 0.04, 0.87, 0.17]); R = len(self.names)
        self.img = self.sax.imshow(np.ones((R, 2 * self.half, 3)), aspect="auto", interpolation="nearest", extent=[-10, 10, R - 0.5, -0.5])
        self.sax.axvline(0, color="k", lw=1.5); self.sax.set_yticks(range(R)); self.sax.set_yticklabels(self.names, fontsize=9)
        self.sax.set_xlabel("seconds from this frame", fontsize=8); self.sax.tick_params(labelsize=8)
        self.home = self.ax.scatter([], [], c=HOME_COLOUR, s=90, edgecolors="k", zorder=3); self.away = self.ax.scatter([], [], c=AWAY_COLOUR, s=90, edgecolors="k", zorder=3)
        self.ball = self.ax.scatter([], [], c="w", s=55, edgecolors="k", zorder=4)
        self.ghost = self.ax.scatter([], [], facecolors="none", edgecolors="k", linestyles=":", s=55, zorder=4)   # true ball where the model's ball is hidden (DFL)
        self.txt = self.ax.text(0.01, 0.98, "", transform=self.ax.transAxes, va="top", fontsize=13, family="monospace", bbox=dict(fc="w", alpha=0.85))

    def frame(self, d, i, title, cam=None):
        p, t, b = d["players"][i], d["team"], d["ball"][i]; none = np.empty((0, 2))
        for side, art in ((0, self.home), (1, self.away)):
            m = (t == side) & np.isfinite(p[:, 0]); art.set_offsets(p[m, :2] if m.any() else none)
        self.ball.set_offsets(b[None, :2] if np.isfinite(b[0]) else none)
        bt = d.get("ball_true"); hidden = bt is not None and not np.isfinite(b[0]) and np.isfinite(bt[i, 0])
        self.ghost.set_offsets(bt[i][None, :2] if hidden else none)
        F = len(next(iter(self.strips.values()))); lo, hi = i - self.half, i + self.half; s0, s1 = max(lo, 0), min(hi, F)
        im = np.ones((len(self.names), 2 * self.half, 3))
        for r, k in enumerate(self.names): im[r, s0 - lo:s1 - lo] = self.cols[k][s0:s1]
        self.img.set_data(im); self.txt.set_text("  ".join(f"{k}={self.strips[k][i]}" for k in self.names))
        self.ax.set_title(f"{title}   frame {i}  {i // FPS // 60:02d}:{i // FPS % 60:02d}", fontsize=10)
        self.fig.canvas.draw(); out = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        if cam is None: return out
        from PIL import Image
        ph = cam.shape[0]; out = np.asarray(Image.fromarray(out).resize((int(out.shape[1] * ph / out.shape[0]), ph)))
        return np.hstack([cam, out])

    def close(self): self.plt.close(self.fig)

def render(pitch, d, a0, a1, title, writer, step=1, cams=None):
    for i in range(a0, a1, step):
        cam = next(cams, None) if cams is not None else None
        if cams is not None and cam is None: break
        writer.write(pitch.frame(d, i, title, cam))


# ---------------------------------------------------------------------------------------------- commands

def cmd_dfl(a):
    from src.seq import decode            # BiGRU decode exactly as the submission (Viterbi switch 3 + 3-s minimum dead)
    from src.possession import score
    z = np.load(SEQ_FOLDS, allow_pickle=True); labs, lps = list(z["labels"]), list(z["logp"]); assert len(labs) == len(IDS), len(labs)
    os.makedirs(OUT, exist_ok=True); allev, table = [], []
    for k, mid in enumerate(IDS):
        g = np.load(f"{EXT}/dfl_{mid}.npz")["labels"].astype(str)
        assert np.array_equal(np.asarray(labs[k]).astype(str), g), f"seq_folds fold {k} is not {mid}: regenerate with src/seq.py --cv"
        bigru = decode(lps[k]).astype(str); f = os.path.join(OUT, f"pathcrf_{a.base + k}_{mid}.npy")
        if os.path.exists(f): pc = np.load(f).astype(str); assert len(pc) == len(g), (mid, len(pc), len(g))
        else: pc = np.full(len(g), "N"); print(f"{mid}: {f} missing -> PathCRF columns empty (src/pathcrf_run.py --trial {a.base + k} --dfl {mid} --save)")
        cov = np.isin(pc, ["H", "A"])
        full = bigru.copy(); m = cov & (bigru != "D"); full[m] = pc[m]        # the submission's rule: PathCRF team on frames our model calls alive
        gtdead = g.copy(); m = cov & (g != "D"); gtdead[m] = pc[m]           # src/pathcrf_run.py --dfl: GT dead balls, GT team where PathCRF has no edge
        np.savez(os.path.join(OUT, f"oof_{mid}.npz"), gt=g, full=full, gtdead=gtdead, bigru=bigru, pathcrf=pc)
        r = {"match": mid, "coverage": cov[g != "D"].mean()}
        for name, p in (("full", full), ("bigru", bigru), ("pathcrf_gtdead", gtdead)): r[name] = score(g, p)
        for o in ("onset", "dead", "team"): r["fix_" + o] = score(g, oracle(g, full, o))
        ev = classify(g, full, mid); allev += ev; table.append(r)
        timeline(f"DFL {mid} (out of fold)", {"GT": g, "pipeline": full, "BiGRU": bigru, "PathCRF": pc}, ev, os.path.join(OUT, f"timeline_{mid}.png"))
    cols = ("full", "bigru", "pathcrf_gtdead", "fix_onset", "fix_dead", "fix_team")
    print("\nspellF1 / macroF1, out of fold. full = submission rule (BiGRU dead balls + PathCRF team); fix_* = full with one error source taken from GT")
    print(f"{'match':8s} {'cov':>5s} " + " ".join(f"{c:>14s}" for c in cols))
    for r in table: print(f"{r['match']:8s} {r['coverage']:5.2f} " + " ".join(f"{r[c]['spell_f1_mean']:6.3f}/{r[c]['macro_f1']:.3f}" for c in cols))
    print(f"{'mean':8s} {np.mean([r['coverage'] for r in table]):5.2f} " + " ".join(f"{np.mean([r[c]['spell_f1_mean'] for r in table]):6.3f}/{np.mean([r[c]['macro_f1'] for r in table]):.3f}" for c in cols))
    print("\nerror types over the 7 matches (full pipeline): count, total seconds, median seconds")
    for t in TYPES:
        s = [e["seconds"] for e in allev if e["type"] == t]
        if s: print(f"  {t:18s} {len(s):5d} {sum(s):9.0f} {np.median(s):7.1f}")
    with open(os.path.join(OUT, "errors.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["match", "type", "start", "stop", "seconds", "detail"]); w.writeheader(); w.writerows(allev)
    print("wrote", os.path.join(OUT, "errors.csv"))
    print("caveats: PathCRF ran on clean DFL tracking (BAR-ATH's is degraded), so 'full' is optimistic; pathcrf_gtdead also keeps GT team where PathCRF has no edge.")

def cmd_clips(a):
    from src.v2 import dfl_game           # seed 99 = the degraded held-out match the BiGRU folds saw
    ev = [e for e in csv.DictReader(open(os.path.join(OUT, "errors.csv"))) if e["type"] == a.type and (not a.match or e["match"] == a.match)]
    ev.sort(key=lambda e: -float(e["seconds"])); cache = {}
    for e in ev[:a.n]:
        mid, s, t = e["match"], int(e["start"]), int(e["stop"])
        if mid not in cache:
            z = np.load(os.path.join(OUT, f"oof_{mid}.npz")); d = dfl_game(mid, 99)
            cache[mid] = (d, Pitch(tuple(np.asarray(d.get("pitch", (105.0, 68.0)), float)), {"GT": z["gt"], "ours": z["full"], "BiGRU": z["bigru"], "PathCRF": z["pathcrf"]}))
        d, pitch = cache[mid]; a0 = max(0, s - 5 * FPS); a1 = min(len(d["ball"]), max(t, s + 2 * FPS) + 5 * FPS, a0 + a.max_seconds * FPS)
        w = Writer(os.path.join(OUT, "clips", f"{e['type']}_{mid}_{s}.mp4"), FPS / a.step)
        render(pitch, d, a0, a1, f"{mid} {e['type']} {e['detail']}", w, a.step); w.close()
    if not ev: print("no events of type", a.type, "- run `eda/qa.py dfl` first; types:", ", ".join(TYPES))

def cmd_pff(a):
    z = np.load(f"{EXT}/pff_{a.gid}.npz", allow_pickle=True); rule = z["labels"].astype(str)
    t = np.load(f"{EXT}/pff_labels2_{a.gid}.npy", allow_pickle=True); teacher = LAB[t] if t.dtype.kind in "iu" else t.astype(str)
    assert len(teacher) == len(rule), f"teacher labels {len(teacher)} frames vs rule {len(rule)}: not on the same frames"
    for name, l in (("rule", rule), ("teacher", teacher)):
        sp = [r for r in runs(l) if r[2] in ("H", "A")]
        print(f"{name:8s} dead {np.mean(l == 'D'):.2f}  stoppages {sum(r[2] == 'D' for r in runs(l)):4d}  possessions {len(sp):4d}  mean possession {np.mean([e - s for s, e, _ in sp]) / FPS:5.1f}s")
    alive_both = (rule != "D") & (teacher != "D")
    print(f"frame agreement {np.mean(rule == teacher):.3f}; team agreement where both alive {np.mean(rule[alive_both] == teacher[alive_both]):.3f}")
    dis = (rule == "D") != (teacher == "D")
    seg = sorted([(e - s, s, e) for s, e, v in runs(dis) if v == "True" and e - s >= 2 * FPS], reverse=True)
    print(f"dead/alive disagreements >= 2 s: {len(seg)}, {sum(x[0] for x in seg) / FPS / 60:.1f} min; rule says dead in {sum(rule[s] == 'D' for _, s, _ in seg)} of them")
    vis = np.where(np.isfinite(z["ball"][:, 0]), "V", "N")
    timeline(f"PFF {a.gid}: rule vs teacher labels", {"rule": rule, "teacher": teacher, "ball seen": vis},
             [dict(type="spurious_stoppage" if rule[s] == "D" else "missed_stoppage", start=s) for _, s, _ in seg], os.path.join(OUT, f"pff_{a.gid}.png"))
    print("  (ticks: purple = rule says dead, teacher alive; black = teacher says dead, rule alive)")
    if a.clips:
        d = {"players": z["players"], "ball": z["ball"], "team": z["team"]}; pitch = Pitch((105.0, 68.0), {"rule": rule, "teacher": teacher})
        for n, s, e in seg[:a.clips]:
            a0 = max(0, s - 5 * FPS); w = Writer(os.path.join(OUT, "clips", f"pff_{a.gid}_{s}.mp4"), FPS / a.step)
            render(pitch, d, a0, min(len(rule), e + 5 * FPS, a0 + a.max_seconds * FPS), f"PFF {a.gid} rule={rule[s]} teacher={teacher[s]} ({n / FPS:.0f}s)", w, a.step); w.close()

def barath_checks(lab, gt):
    """Pipeline-bug checks on a submission CSV; no ground truth beyond the public window is involved."""
    ok = True
    for h, (s, e) in enumerate(GRADED, 1):
        l = lab[s:e]; sp = [r for r in runs(l) if r[2] in ("H", "A")]; dd = [e2 - s2 for s2, e2, v in runs(l) if v == "D"]
        print(f"half {h}: H {np.mean(l == 'H'):.2f} A {np.mean(l == 'A'):.2f} D {np.mean(l == 'D'):.2f} X {np.mean(l == 'X'):.3f} | "
              f"possessions {len(sp)}, stoppages {len(dd)}, longest stoppage {max(dd, default=0) / FPS:.0f}s, longest possession {max([e2 - s2 for s2, e2, _ in sp], default=0) / FPS:.0f}s")
        if (l == "X").any(): print(f"  !! {int((l == 'X').sum())} X frames inside graded half {h}"); ok = False
    for name, (s, e) in (("before kick-off", (0, GRADED[0][0])), ("half-time", (GRADED[0][1], GRADED[1][0]))):
        if (lab[s:e] == "D").mean() < 0.5 and (lab[s:e] == "X").mean() < 0.5: print(f"  !! {name} is mostly H/A"); ok = False
    w = slice(11185, 18685); m = np.isin(gt[w], ["H", "A", "D"])
    print(f"public window (sanity only): frame agreement {np.mean(lab[w][m] == gt[w][m]):.3f}")
    print("checks:", "ok" if ok else "SEE !! ABOVE")

def cmd_barath(a):
    from eda.atd import load; from games import GAMES, GAME
    lab = np.loadtxt(os.path.join(ROOT, a.csv), dtype=str, delimiter=",", skiprows=1, usecols=1)
    gt = np.loadtxt(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"), dtype=str, delimiter=",", skiprows=1, usecols=1)
    assert len(lab) == len(gt) == 199319, (len(lab), len(gt)); barath_checks(lab, gt)
    if a.checks_only: return
    graded = np.zeros(len(lab), bool)
    for s, e in GRADED: graded[s:e] = True
    ch = [u for u in np.flatnonzero(lab[1:] != lab[:-1]) + 1 if graded[u] and graded[u - 1]]
    wins = []
    for u in ch:                                                      # 2 s before, 3 s after each change; overlapping windows merge
        s, e = u - 2 * FPS, u + 3 * FPS
        if wins and s <= wins[-1][1]: wins[-1][1] = e
        else: wins.append([s, e])
    wins = wins[:a.max] if a.max else wins
    print(f"{len(ch)} label changes on graded frames -> {len(wins)} windows, {sum(e - s for s, e in wins) / FPS / 60:.1f} min of video")
    d = load("tactical"); pitch = Pitch(GAMES[GAME]["pitch"], {"ours": lab, "public GT": gt}); w = Writer(os.path.join(OUT, "barath_changes.mp4"), FPS / a.step)
    for k, (s, e) in enumerate(wins, 1):
        cams = camera(d, s, e, a.step) if a.video else None
        render(pitch, d, s, e, f"BAR-ATH window {k}/{len(wins)}", w, a.step, cams)
    w.close()

def camera(d, a0, a1, step):
    """Tactical camera frames a0..a1 (every step-th) with the tracking projected on them, 960x540."""
    from eda.render import video_frames, overlay; from games import asset_paths
    paths = asset_paths(source="tactical"); Hs = np.loadtxt(paths["homography"], delimiter=",", skiprows=1)[:, 1:10].reshape(-1, 3, 3)
    for k, img in enumerate(video_frames(paths["video"], a0, a1 - a0, 960, 540)):
        if k % step == 0: yield overlay(img, d, a0 + k, Hs[a0 + k], 960 / 1920)

def cmd_selftest(a):
    s = lambda x: np.array(list(x))
    g = s("HHHHHHHHHH" * 10 + "D" * 100 + "AAAAAAAAAA" * 10 + "HHHHHHHHHH" * 10)       # H 100, stoppage 100, A 100, live change, H 100
    p = s("HHHHHHHHHH" * 15 + "D" * 50 + "AAAAAAAAAA" * 10 + "AAAAAAAAAA" * 10)        # stoppage 2 s late, change missed (never)
    t = {e["type"]: e for e in classify(g, p)}
    assert set(t) == {"dead_onset_late", "missed_change", "wrong_team_spell"}, t
    assert t["dead_onset_late"]["start"] == 100 and t["dead_onset_late"]["stop"] == 150 and t["dead_onset_late"]["seconds"] == 2.0
    assert t["missed_change"]["start"] == 300 and "never" in t["missed_change"]["detail"]
    assert "".join(oracle(g, p, "onset")) == "".join(g[:300]) + "A" * 100
    assert "".join(oracle(g, p, "both")) == "".join(g)
    assert "".join(oracle(g, p, "team")) == "".join(p[:300]) + "H" * 100
    q = s("H" * 100 + "D" * 100 + "A" * 100 + "H" * 100); q[350:400] = "D"                # spurious stoppage inside the A possession
    assert {e["type"] for e in classify(g, q)} == {"spurious_stoppage"}, classify(g, q)
    r = s("H" * 100 + "D" * 100 + "A" * 100 + "H" * 100); r[150:160] = "H"               # a 0.4 s blip of H inside the stoppage
    assert {e["type"] for e in classify(g, r)} <= {"restart_early", "dead_onset_late"} or classify(g, r) == [], classify(g, r)
    late = s("H" * 100 + "D" * 100 + "A" * 100 + "A" * 40 + "H" * 60)                   # change 1.6 s late: inside tolerance
    assert classify(g, late) == [], classify(g, late)
    later = s("H" * 100 + "D" * 100 + "A" * 100 + "A" * 75 + "H" * 25)                  # 3 s late: missed, lag reported
    e = [x for x in classify(g, later) if x["type"] == "missed_change"]; assert e and e[0]["seconds"] == 3.0, classify(g, later)
    assert "".join(oracle(g, s("H" * 100 + "D" * 150 + "A" * 150), "dead")) == "H" * 100 + "D" * 100 + "H" * 50 + "A" * 150
    print("classify / oracle: ok")
    if a.render:                                                       # synthetic match through the renderer and the timeline
        rng = np.random.default_rng(0); F = 600; pl = np.cumsum(rng.normal(0, 0.1, (F, 22, 3)), 0) + np.r_[rng.uniform(10, 95, (22, 1)), rng.uniform(5, 63, (22, 1)), np.zeros((22, 1))].reshape(3, 22).T
        b = pl[:, 3].copy(); b[200:260] = np.nan; d = {"players": pl.astype(np.float32), "ball": b, "ball_true": pl[:, 3], "team": np.r_[np.zeros(11), np.ones(11)].astype(np.int8)}
        gg = np.r_[g, g[:200]]; pp = np.r_[p, p[:200]]
        timeline("selftest", {"GT": gg, "ours": pp}, classify(gg, pp, "x"), os.path.join(OUT, "selftest_timeline.png"), panel_min=0.1)
        w = Writer(os.path.join(OUT, "selftest_clip.mp4"), FPS / 2); pitch = Pitch((105.0, 68.0), {"GT": gg, "ours": pp}); render(pitch, d, 150, 350, "selftest", w, 2); w.close()
        print("render: ok")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dfl"); p.add_argument("--base", type=int, default=200, help="PathCRF trial of fold 0 (folds are base+k)")
    p = sub.add_parser("clips"); p.add_argument("--type", required=True, choices=list(TYPES)); p.add_argument("--n", type=int, default=5); p.add_argument("--match")
    p.add_argument("--step", type=int, default=1); p.add_argument("--max-seconds", type=int, default=40)
    p = sub.add_parser("pff"); p.add_argument("gid"); p.add_argument("--clips", type=int, default=0); p.add_argument("--step", type=int, default=1); p.add_argument("--max-seconds", type=int, default=40)
    p = sub.add_parser("barath"); p.add_argument("--csv", default="evaluation/results/pathcrf.csv"); p.add_argument("--video", action="store_true", help="tactical camera beside the pitch")
    p.add_argument("--checks-only", action="store_true", help="sanity checks, no rendering"); p.add_argument("--step", type=int, default=2); p.add_argument("--max", type=int, default=0)
    p = sub.add_parser("selftest"); p.add_argument("--render", action="store_true")
    a = ap.parse_args(); {"dfl": cmd_dfl, "clips": cmd_clips, "pff": cmd_pff, "barath": cmd_barath, "selftest": cmd_selftest}[a.cmd](a)
