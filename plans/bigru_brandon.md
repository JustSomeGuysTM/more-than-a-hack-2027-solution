# BiGRU (dead balls) — Brandon

Second job after the submission checklist (`plans/submission_checklist.md` comes first: a file the repo cannot
regenerate scores nothing). Model changes must be decided by **Nov 8** so the clean-clone reproduction can be redone.

## What the BiGRU does in the submission

The submission is two models glued together (`src/pathcrf_run.py`, function `main`):
1. **BiGRU** (`src/seq.py`) reads 30-s windows of per-frame features (`src/features.py`: ball position and speed,
   nearest players, team shapes, speeds) and outputs P(H), P(A), P(D) per frame. A Viterbi pass with a switch penalty
   turns that into labels, then every dead stretch shorter than 3 s is removed (`possession.min_dead`).
2. **PathCRF** (trial 401) decides the team on every frame the BiGRU calls alive, but only inside alive stretches of
   at least 10 s (its window). On shorter stretches the BiGRU's own team call is used.

So the BiGRU owns: (a) dead vs alive everywhere, (b) where PathCRF's episodes start and end, (c) the team on the ~4%
of alive frames in short stretches.

Training data: the 7 public DFL (Bundesliga) matches with official labels, with the ball degraded to look like our
tracker's (`src/external.py`: same gap lengths, a dead ball that keeps moving, missing players). Never BAR-ATH labels.

## What QA says it is worth (`eda/NOTES.md`, "QA" section)

| DFL out of fold, mean of 7 | spellF1 / macroF1 |
|---|---|
| pipeline as submitted | 0.589 / 0.887 |
| dead balls made perfect | 0.615 / 0.920 |
| only stoppage onsets fixed | 0.596 / 0.894 |
| team made perfect | 0.904 / 0.954 |

The dead-ball ceiling is about +0.03 spellF1 (and +0.03 macroF1, which is the tie-break). The short-stretch fallback
is wrong 30% of the time vs 9% for PathCRF, and causes 12% of all wrong-team frames. Those are the two targets.

## How to run it

```
.venv/bin/python src/seq.py --cv        # leave-one-match-out on DFL, a few hours on CPU/MPS; prints spellF1 per switch
.venv/bin/python src/seq.py --train     # fit on all 7 -> src/seq_model.pt
.venv/bin/python src/seq.py             # BAR-ATH -> evaluation/results/seq.csv (the dead balls the submission uses)
```
`--cv` overwrites `eda/out/seq_folds.npz`, which `eda/qa.py` and `src/pathcrf_run.py --oof` read. Copy it first. To see
the effect on the whole pipeline: rerun `src/pathcrf_run.py --trial 20k --dfl <match> --save --oof` for the 7 folds (PathCRF
venv, ~2 min each; loop in the `eda/qa.py` docstring), then `.venv/bin/python eda/qa.py dfl` and read the `full` column.
Current numbers to beat: BiGRU 0.464 / 0.839 alone, pipeline 0.589 / 0.887.

## Ideas (the finance analogies are real ones)

1. **It is a regime-switching model.** Alive/dead is a two-regime process; the Viterbi switch penalty is a crude
   transition matrix and `min_dead` a crude duration prior. Fit a hidden semi-Markov model instead: duration
   distributions for stoppages and for live play estimated from the DFL labels (stoppages are 2-60 s with a long tail),
   asymmetric entry/exit costs (`switch` vs `switch_d` already exist; they are tied at 3 now). Cheap, interpretable, no
   retraining of the net: only `decode` changes, so it can be tuned on the cached fold log-probs in minutes.
2. **Calibrate, then decode.** `temp` is 1.0. Check reliability of P(D) per fold (as you would for a probability
   forecast) and fit temperature or isotonic per class on the training folds only.
3. **Bag it.** Average log-probs over 3-5 seeds. Variance reduction for free; the folds show ±0.03 match to match.
4. **Longer context.** `WIN = 750` is 30 s. Stoppages can be longer than the window; try 60 s, or a dilated TCN.
5. **Better fallback for short stretches.** A small head trained only on alive stretches < 10 s (restarts: throw-ins,
   free kicks), where the team that restarts usually keeps the ball.
6. **Features that look like stoppages** (`src/features.py`): everyone's speed collapsing, players clustering in the box
   (corners), a wall forming, the ball near a line and still. Feature changes need `--cv` reruns.

Rules: decide on DFL leave-one-out only; the public five minutes can veto, never select. No labels made by a person on
our match. Write each result (even negative) as a short paragraph in `eda/NOTES.md`.
