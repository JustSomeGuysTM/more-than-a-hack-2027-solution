"""Commentary -> stoppage probability, trained on OTHER games (SoccerNet-Echoes transcripts +
SoccerNet action labels), applied to our broadcast transcript through the clock alignment.

    .venv/bin/python src/commentary_model.py --train      # TF-IDF + logistic regression, game-held-out AUC -> src/commentary_model.pkl
    .venv/bin/python src/commentary_model.py              # BAR-ATH -> eda/out/commentary_stoppage.npz (per tactical frame P(stoppage))

Window = WIN s of transcript text; label = a stoppage event (ball out, throw-in, corner, foul, free
kick, kick-off, offside, goal, card, substitution) inside [t0 - LAG_BEFORE, t1 + LAG_AFTER], since
commentators mention things a few seconds after they happen.
"""
import argparse, glob, json, os, pickle, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
SN = os.path.join(ROOT, "data_raw/external/soccernet"); MODEL = os.path.join(ROOT, "src/commentary_model.pkl")
WIN, STRIDE, LAG_BEFORE, LAG_AFTER = 10.0, 5.0, 2.0, 4.0
STOP = {"Ball out of play", "Throw-in", "Corner", "Foul", "Indirect free-kick", "Direct free-kick", "Kick-off", "Offside", "Goal", "Yellow card", "Red card", "Yellow->red card", "Substitution", "Penalty"}

def echoes():
    import pyarrow as pa
    f = f"{SN}/SN-echoes/whisper_v3_en/1.0.0/soccer_net_echoes_hf_dataset-train.arrow"
    try: t = pa.ipc.open_stream(f).read_all().to_pandas()
    except Exception: t = pa.ipc.open_file(f).read_all().to_pandas()
    t["match"] = t.game.str.rsplit("/", n=1).str[0]; t["half"] = t.game.str[-1].astype(int); return t

def labels():
    out = {}
    for f in glob.glob(f"{SN}/SN-Labels/**/Labels-v2.json", recursive=True):
        m = os.path.relpath(os.path.dirname(f), f"{SN}/SN-Labels"); ev = []
        for a in json.load(open(f))["annotations"]:
            if a["label"] in STOP: h, mmss = a["gameTime"].split(" - "); mi, se = mmss.split(":"); ev.append((int(h), int(mi) * 60 + int(se)))
        out[m] = ev
    return out

def windows(seg):
    """seg: DataFrame of one half (start_time, end_time, text) -> list of (t0, t1, text)."""
    if seg.empty: return []
    T = seg.end_time.max(); out = []
    for t0 in np.arange(0, T, STRIDE):
        s = seg[(seg.end_time > t0) & (seg.start_time < t0 + WIN)]
        if len(s): out.append((t0, t0 + WIN, " ".join(s.text)))
    return out

def train():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score
    ec, lab = echoes(), labels(); rows = []
    for (m, h), seg in ec.groupby(["match", "half"]):
        if m not in lab: continue
        ev = np.array([t for hh, t in lab[m] if hh == h])
        for t0, t1, txt in windows(seg): rows.append((m, txt, int(((ev >= t0 - LAG_BEFORE) & (ev <= t1 + LAG_AFTER)).any()) if len(ev) else 0))
    df = pd.DataFrame(rows, columns=["match", "text", "y"]); print("windows", len(df), "matches", df.match.nunique(), "positive rate %.2f" % df.y.mean())
    games = df.match.unique(); rng = np.random.default_rng(0); test = set(rng.choice(games, max(1, len(games) // 5), replace=False))
    tr, te = df[~df.match.isin(test)], df[df.match.isin(test)]
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200000, sublinear_tf=True), LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced"))
    clf.fit(tr.text, tr.y); p = clf.predict_proba(te.text)[:, 1]
    print("held-out games: AUC %.3f  AP %.3f (base %.2f)" % (roc_auc_score(te.y, p), average_precision_score(te.y, p), te.y.mean()))
    v, lr = clf.named_steps["tfidfvectorizer"], clf.named_steps["logisticregression"]; names = np.array(v.get_feature_names_out()); top = np.argsort(lr.coef_[0])[::-1][:25]
    print("top stoppage terms:", ", ".join(names[top]))
    clf.fit(df.text, df.y); pickle.dump(clf, open(MODEL, "wb")); print("->", MODEL)

def run_barath():
    from eda.align import b2t
    clf = pickle.load(open(MODEL, "rb")); segs = [json.loads(l) for l in open(os.path.join(ROOT, "eda/audio/commentary_broadcast.jsonl"))]
    seg = pd.DataFrame({"start_time": [s["start"] for s in segs], "end_time": [s["end"] for s in segs], "text": [s["text"] for s in segs]})
    F = 199319; prob = np.full(F, np.nan, np.float32)
    for t0, t1, txt in windows(seg):
        p = clf.predict_proba([txt])[0, 1]; a, b = b2t(t0 + LAG_BEFORE), b2t(t1 - 1)       # the window talks about play from ~t0-lag to t1
        if a is None or b is None: continue
        prob[max(a - int(LAG_AFTER * 25), 0):b] = np.fmax(prob[max(a - int(LAG_AFTER * 25), 0):b], p)
    out = os.path.join(ROOT, "eda/out/commentary_stoppage.npz"); np.savez_compressed(out, prob=prob); print("covered frames %.2f" % np.isfinite(prob).mean(), "->", out)
    gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values; w = slice(11185, 18685); D = gt[w] == "D"; pw = prob[w]
    for q in (0.5, 0.7, 0.85): m = pw >= q; print(f"public window sanity: P(D | prob>={q}) = {D[m].mean():.2f} on {m.sum()} frames; base P(D) = {D.mean():.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", action="store_true"); a = ap.parse_args(); train() if a.train else run_barath()
