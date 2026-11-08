"""Sequence model over the per-frame features: BiGRU trained on DFL (ball degraded, H/A mirrored),
log-softmax outputs as Viterbi emissions. Learns possession boundaries from context rather than
from hand-made rolling features.

    .venv/bin/python src/seq.py --cv [--epochs 8]    # leave-one-match-out on DFL
    .venv/bin/python src/seq.py --train              # fit on all 7 -> src/seq_model.pt
    .venv/bin/python src/seq.py                      # BAR-ATH -> evaluation/results/seq.csv (+ log-probs for --blend)
    .venv/bin/python src/seq.py --blend [--trial 401] [--out ...]   # after src/pathcrf_run.py --trial N: soft PathCRF team vote -> evaluation/results/pathcrf.csv
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.features import features
from src.possession import viterbi, score, LAB, min_dead, in_play
from src.v2 import IDS, dfl_game, mirror
import torch, torch.nn as nn
MODEL = os.path.join(ROOT, "src/seq_model.pt"); WIN = 750; HID = 64
FOLDS = os.path.join(ROOT, f"eda/out/seq_folds{os.environ.get('TAG', '')}.npz"); HGB_FOLDS = os.path.join(ROOT, "eda/out/dead_folds_carry2.npz")   # eda/dead_diag.py TAG=_carry2   # 30 s windows at 25 fps
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BARATH_LP = os.path.join(ROOT, "eda/out/seq_barath_logp.npy"); PC_TEAM = os.path.join(ROOT, "eda/out/pathcrf_barath_team_{trial}.npy")   # local only (BAR-ATH-derived); one per PathCRF trial
TRIAL = 401   # the submitted PathCRF trial, as in src/pathcrf_run.py and pod/pathcrf_setup.sh
BLEND_W = 4.0   # PathCRF's team vote in soft_team: DFL leave-one-out 0.602 vs 0.588 hard, better on 7/7 folds (eda/team_blend.py, 2026-11-04)
S = dict(temp=1.0, switch=3.0, switch_d=3.0)   # DFL leave-one-out, 8 epochs (pod, 2026-10-27): switch 3 -> spellF1 0.603 / macroF1 0.914, 6 -> 0.598, 10 -> 0.590, 15 -> 0.577

class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__(); self.norm = nn.BatchNorm1d(n_in); self.gru = nn.GRU(n_in, HID, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1); self.out = nn.Linear(2 * HID, 3)
    def forward(self, x):                                  # x (B, T, F)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2); h, _ = self.gru(x); return self.out(h).log_softmax(-1)

def prep(X):
    return torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)

def windows(X, y, stride=WIN // 2):
    for s in range(0, len(X) - WIN + 1, stride): yield X[s:s + WIN], y[s:s + WIN]

def fit(games, epochs, names):
    net = Net(len(names)).to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3); xs, ys = [], []
    for d in games:
        X, _ = features(d, (105.0, 68.0)); y = (d["labels"] == "A") * 1 + (d["labels"] == "D") * 2
        for Xv, yv in ((X, y), (mirror(X, names), np.where(y == 2, 2, 1 - y))):
            for xw, yw in windows(Xv, yv): xs.append(xw); ys.append(yw)
    Xt, yt = prep(np.stack(xs)), torch.tensor(np.stack(ys), dtype=torch.long); n = len(Xt); print("windows", n, flush=True)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, 32):
            b = perm[i:i + 32]; xb, yb = Xt[b].to(DEV), yt[b].to(DEV); opt.zero_grad()
            loss = nn.functional.nll_loss(net(xb).reshape(-1, 3), yb.reshape(-1)); loss.backward(); opt.step(); tot += loss.item() * len(b)
        print(f"epoch {ep + 1}/{epochs} loss {tot / n:.4f}", flush=True)
    return net

@torch.no_grad()
def logp(net, X):
    """Full-sequence inference in overlapping windows, averaging the overlaps."""
    net.eval(); F = len(X); out = np.zeros((F, 3)); cnt = np.zeros(F); Xt = prep(X)
    for s in list(range(0, max(F - WIN, 0) + 1, WIN // 2)) + [max(F - WIN, 0)]:
        out[s:s + WIN] += net(Xt[s:s + WIN][None].to(DEV))[0].cpu().numpy(); cnt[s:s + WIN] += 1
    return out / np.maximum(cnt, 1)[:, None]

def decode(lp, s=S): return min_dead(LAB[viterbi(s["temp"] * lp, s["switch"], s["switch_d"])])

def cv(epochs):
    folds = []
    for held in IDS:
        games = [dfl_game(m, k) for k, m in enumerate(IDS) if m != held]; _, names = features(games[0], (105.0, 68.0))
        net = fit(games, epochs, names); d = dfl_game(held, 99); X, _ = features(d, (105.0, 68.0)); lp = logp(net, X)
        folds.append((d["labels"], lp)); print(held, "frame acc %.3f" % (LAB[lp.argmax(1)] == d["labels"]).mean(), flush=True)
    np.savez(FOLDS, labels=np.array([l for l, _ in folds], dtype=object), logp=np.array([lp for _, lp in folds], dtype=object))
    report("BiGRU", folds)
    if os.path.exists(HGB_FOLDS):                                 # same held-out matches, same seed 99 -> same augmented ball
        z = np.load(HGB_FOLDS, allow_pickle=True); hgb = list(z["logp"])
        report("BiGRU H/A + HGB D", [(l, combine(lp, h)) for (l, lp), h in zip(folds, hgb)])
        report("mean log-prob", [(l, (lp + h) / 2) for (l, lp), h in zip(folds, hgb)])

def report(name, folds):
    for sw in (3, 6, 10, 15):
        r = [score(l, decode(lp, dict(S, switch=sw, switch_d=sw))) for l, lp in folds]
        print(f"{name:18s} switch={sw:2d}: spellF1={np.mean([x['spell_f1_mean'] for x in r]):.3f} macroF1={np.mean([x['macro_f1'] for x in r]):.3f} spells={np.mean([x['n_pred_spells'] for x in r]):.0f}")

def combine(lp_seq, lp_hgb):
    """BiGRU decides H vs A, HGB decides alive vs dead: P(D)=P_hgb(D), P(H)=P_hgb(alive)*P_seq(H|alive)."""
    alive = np.logaddexp(lp_hgb[:, 0], lp_hgb[:, 1]); ha = lp_seq[:, :2] - np.logaddexp(lp_seq[:, 0], lp_seq[:, 1])[:, None]
    return np.column_stack([alive[:, None] + ha, lp_hgb[:, 2]])

def soft_team(lp, dead, pc, w=BLEND_W, s=S):
    """Dead balls kept exactly (dead = label array, "D" = dead); on the other frames a Viterbi over H/A only, with the BiGRU's
    team log-prob plus w for PathCRF's team (pc in H/A; anything else = not covered, no vote). A confident BiGRU can overrule it."""
    a = dead != "D"; e = np.full_like(lp, -20.0); e[~a, 2] = 0
    e[a, :2] = lp[a, :2] - np.logaddexp(lp[a, 0], lp[a, 1])[:, None]; e[a & (pc == "H"), 0] += w; e[a & (pc == "A"), 1] += w
    p = decode(e, s).astype(object); p[~a] = "D"; return p

def blend(seq_csv, out, trial=TRIAL):
    import pandas as pd
    f = PC_TEAM.format(trial=trial)
    if not os.path.exists(f): sys.exit(f"{f} missing: run src/pathcrf_run.py --trial {trial} first")
    lab = pd.read_csv(seq_csv)["label"].values.astype(str); lp = np.load(BARATH_LP); pc = np.load(f).astype(str)
    assert ((decode(lp) == "D") == (lab == "D")).all(), f"dead balls in {seq_csv} do not match {BARATH_LP}: rerun src/seq.py (default switch)"
    p = soft_team(lp, lab, pc); pd.DataFrame({"frame": np.arange(len(p)), "label": p}).to_csv(out, index=False)
    print(f"PathCRF trial {trial} ({os.path.basename(f)})")
    gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
    a = lab != "D"; print("changed vs hard PathCRF on alive frames: %.3f" % (p[a & np.isin(pc, ["H", "A"])] != pc[a & np.isin(pc, ["H", "A"])]).mean())
    print("BAR-ATH public window (sanity only):", score(gt, p)); print("wrote", out)

def train(epochs):
    games = [dfl_game(m, k) for k, m in enumerate(IDS)]; _, names = features(games[0], (105.0, 68.0)); net = fit(games, epochs, names)
    torch.save({"state": net.state_dict(), "names": names}, MODEL); print("->", MODEL)

def run_barath(out, switch=S["switch"]):
    import pandas as pd; from eda.atd import load; from games import GAMES, GAME
    m = torch.load(MODEL, map_location="cpu"); d = load("tactical"); d["fps"] = 25; F = len(d["ball"])
    X, names = features(d, GAMES[GAME]["pitch"]); assert names == m["names"]; net = Net(len(names)).to(DEV); net.load_state_dict(m["state"])
    lp = logp(net, X); lp[~in_play(F), :2] = -20   # outside play is dead by definition (src/possession.play_windows)
    os.makedirs(os.path.dirname(BARATH_LP), exist_ok=True); np.save(BARATH_LP, lp)
    lab = decode(lp, dict(S, switch=switch, switch_d=switch)); pd.DataFrame({"frame": np.arange(F), "label": lab}).to_csv(out, index=False)
    gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
    print("BAR-ATH public window (sanity only):", score(gt, lab)); print("wrote", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cv", action="store_true"); ap.add_argument("--train", action="store_true"); ap.add_argument("--blend", action="store_true"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", help="default evaluation/results/seq.csv, or evaluation/results/pathcrf.csv with --blend"); ap.add_argument("--switch", type=float, default=S["switch"])
    ap.add_argument("--trial", type=int, default=TRIAL, help="with --blend: the PathCRF trial whose team calls are blended (src/pathcrf_run.py --trial)"); a = ap.parse_args()
    res = lambda name: os.path.join(ROOT, "evaluation/results", name)
    if a.blend: blend(res("seq.csv"), a.out or res("pathcrf.csv"), a.trial)
    else: cv(a.epochs) if a.cv else train(a.epochs) if a.train else run_barath(a.out or res("seq.csv"), a.switch)
