"""Run a PathCRF checkpoint on BAR-ATH (data_raw/pathcrf/BARATH.parquet from src/pathcrf_io.py).

    ~/pathcrf/.venv/bin/python src/pathcrf_run.py [--trial 140] [--alive evaluation/results/v2.csv]
    -> evaluation/results/pathcrf.csv : PathCRF team on alive frames, D from --alive, X for alive frames PathCRF did not cover
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PC = os.path.expanduser("~/pathcrf"); sys.path.insert(0, PC); sys.path.insert(0, ROOT); os.chdir(PC)
from models.utils import build_model, load_trial_args, resolve_model_path
from inference import inference

def main(trial, alive_csv, out):
    args = load_trial_args(f"saved/{trial}"); model = build_model(args, device="cpu")
    model.load_state_dict(torch.load(resolve_model_path(f"saved/{trial}", "state_dict_best_acc.pt"), map_location="cpu", weights_only=False)); model.eval()
    tr = pd.read_parquet(os.path.join(ROOT, "data_raw/pathcrf/BARATH.parquet")); t0 = time.time()
    _, _, micro, stats = inference(model, tr, use_crf=args.get("crf_weight", 0) > 0, decode="indep", window_seconds=args.get("window_seconds"), correct_episode_lasts=False)   # the episode-end fix-up reads padded nodes the model may pick; skip it
    print("inference %.0fs, frames predicted %d of %d" % (time.time() - t0, micro["edge_src"].notna().sum(), len(tr)))
    lab = pd.read_csv(alive_csv)["label"].values.astype(object); F = len(lab); pc = lab.copy()
    print("edge_src prefixes:", micro["edge_src"].astype(str).str.split("_").str[0].value_counts().to_dict())
    team = micro["edge_src"].map(lambda s: "H" if isinstance(s, str) and s.startswith("home") else "A" if isinstance(s, str) and s.startswith("away") else None)   # out_/goal_ nodes: keep the v2 label
    fr = micro.index.values if micro.index.name == "frame_id" else tr["frame_id"].values
    for f, t in zip(fr, team.values):
        if isinstance(t, str) and lab[f] != "D": pc[f] = t          # NaN edge (episode not covered) keeps the v2 label
    pc = np.where(np.isin(pc, ["H", "A", "D", "X"]), pc, "X").astype(object); lab = np.where(np.isin(lab, ["H", "A", "D", "X"]), lab, "X").astype(object)
    print("labels:", {k: int((pc == k).sum()) for k in "HADX"}, "| v2:", {k: int((lab == k).sum()) for k in "HADX"})
    pd.DataFrame({"frame": np.arange(F), "label": pc}).to_csv(out, index=False)
    sys.path.insert(0, os.path.join(ROOT, "evaluation")); import possession_metrics as pm
    gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
    for name, p in (("v2", lab), ("pathcrf+v2 dead", pc)):
        r = pm.evaluate(pm.label_codes(gt), pm.label_codes(p)); print(f"{name:16s} public window: spellF1 {r['spell_f1_mean']:.3f} macroF1 {r['macro_f1']:.3f} spells {r['n_pred_spells']} (gt {r['n_gt_spells']})")
    w = slice(11185, 18685); print("agreement v2 vs pathcrf on alive frames in window: %.3f" % (lab[w][lab[w] != "D"] == pc[w][lab[w] != "D"]).mean())
    print("wrote", out)

def score_external(trial, gid, dfl=False):
    """PathCRF checkpoint on a PFF match (src/pff.py --pathcrf, algorithmic labels) or a DFL match (--dfl, official labels;
    PathCRF's own sportec parquet, rows aligned with data_raw/external/dfl_<id>.npz). Dead frames taken from the labels."""
    from src.pff import pff, PATHCRF; from src.possession import score
    args = load_trial_args(f"saved/{trial}"); model = build_model(args, device="cpu")
    model.load_state_dict(torch.load(resolve_model_path(f"saved/{trial}", "state_dict_best_acc.pt"), map_location="cpu", weights_only=False)); model.eval()
    if dfl: tr = pd.read_parquet(f"{PC}/data/sportec/tracking_processed/{gid}.parquet"); L = np.load(os.path.join(ROOT, f"data_raw/external/dfl_{gid}.npz"))["labels"].astype(object)
    else: tr = pd.read_parquet(f"{PATHCRF}/{gid}.parquet"); L = pff(gid)["labels"].astype(object)
    assert len(tr) == len(L), (len(tr), len(L)); pc = L.copy(); pos = {fid: i for i, fid in enumerate(tr["frame_id"].values)}
    _, _, micro, _ = inference(model, tr, use_crf=args.get("crf_weight", 0) > 0, decode="indep", window_seconds=args.get("window_seconds"), correct_episode_lasts=False)
    fr = micro.index.values if micro.index.name == "frame_id" else tr["frame_id"].values
    for f, s in zip(fr, micro["edge_src"].values):
        f = pos[f]
        if isinstance(s, str) and L[f] != "D" and s[:4] in ("home", "away"): pc[f] = "H" if s[:4] == "home" else "A"
    cov = np.mean([isinstance(s, str) and s[:4] in ("home", "away") for s in micro["edge_src"].values]); r = score(L, pc)
    print(f"PathCRF trial {trial} on {'DFL' if dfl else 'PFF'} {gid}: spellF1 {r['spell_f1_mean']:.3f} macroF1 {r['macro_f1']:.3f} spells {r['n_pred_spells']:.0f} (gt {r['n_gt_spells']:.0f}), frames with a team {cov:.2f}, alive frame acc {(pc[L != 'D'] == L[L != 'D']).mean():.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trial", type=int, default=140); ap.add_argument("--alive", default=os.path.join(ROOT, "evaluation/results/v2.csv")); ap.add_argument("--out", default=os.path.join(ROOT, "evaluation/results/pathcrf.csv")); ap.add_argument("--pff", help="score the checkpoint on this PFF game id instead"); ap.add_argument("--dfl", help="score the checkpoint on this DFL match id (held-out fold) instead")
    a = ap.parse_args(); score_external(a.trial, a.pff) if a.pff else score_external(a.trial, a.dfl, dfl=True) if a.dfl else main(a.trial, a.alive, a.out)
