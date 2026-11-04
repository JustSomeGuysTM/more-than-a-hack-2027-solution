"""Team call: PathCRF (fold trials 200-206, official episodes) blended with HGB / BiGRU fold log-probs on DFL leave-one-out.
    .venv/bin/python eda/team_blend.py hgb:_det0 hgb:_space seq:_base seq:_space
hard = today's proxy: model dead balls, PathCRF team wherever it covers the frame, model team elsewhere.
soft w = team only: dead balls exactly as the model decodes them; on its alive frames a Viterbi over H/A with the model's
team log-prob plus PathCRF's bonus w (model can overrule it). PathCRF's labels here come from official episodes (it only covers
truly alive frames), so a bonus that also raised "alive" would leak the true dead balls: the first version did (0.621, rising with w)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.v2 import IDS, decode as hgb_decode
from src.seq import decode as seq_decode, soft_team
from src.possession import score

def run(tag):
    kind, t = tag.split(":"); z = np.load(os.path.join(ROOT, f"eda/out/{'dead' if kind == 'hgb' else 'seq'}_folds{t}.npz"), allow_pickle=True)
    dec = hgb_decode if kind == "hgb" else seq_decode
    pcs = [np.load(os.path.join(ROOT, f"eda/out/qa/pathcrf_{200 + k}_{m}.npy")).astype(str) for k, m in enumerate(IDS)]
    gts = [np.asarray(l).astype(str) for l in z["labels"]]; lps = list(z["logp"])
    def show(name, preds):
        r = [score(g, p) for g, p in zip(gts, preds)]
        print(f"{tag:12s} {name:10s} spellF1 {np.mean([x['spell_f1_mean'] for x in r]):.3f} macroF1 {np.mean([x['macro_f1'] for x in r]):.3f}", flush=True)
    own = [dec(lp).astype(str) for lp in lps]; show("model", own)
    hard = []
    for d, pc in zip(own, pcs):
        p = d.copy(); m = np.isin(pc, ["H", "A"]) & (d != "D"); p[m] = pc[m]; hard.append(p)
    show("hard", hard)
    for w in (0.5, 1, 2, 4):
        preds = []
        for lp, pc, d in zip(lps, pcs, own):
            if kind == "seq": preds.append(soft_team(lp, d, pc, w).astype(str)); continue
            e = np.full_like(lp, -20.0); a = d != "D"; e[~a, 2] = 0                       # same as soft_team, HGB decoder
            e[a, :2] = lp[a, :2] - np.logaddexp(lp[a, 0], lp[a, 1])[:, None]; e[a & (pc == "H"), 0] += w; e[a & (pc == "A"), 1] += w
            p = dec(e).astype(str); p[~a] = "D"; preds.append(p)
        show(f"soft w={w}", preds)

if __name__ == "__main__":
    for tag in sys.argv[1:]: run(tag)
