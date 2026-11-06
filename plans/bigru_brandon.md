# BiGRU (dead balls) — Brandon Hiles

**Owner:** Brandon Hiles.

Second job after the submission checklist (`plans/submission_checklist.md` comes first: a file the repo cannot
regenerate scores nothing). Model changes must be decided by **Nov 8** so the clean-clone reproduction can be redone.

## What the BiGRU does in the submission

The submission is two models glued together (`src/pathcrf_run.py`, then `src/seq.py --blend`):
1. **BiGRU** (`src/seq.py`) reads 30-s windows of per-frame features (`src/features.py`: ball position and speed,
   nearest players, team shapes, speeds) and outputs P(H), P(A), P(D) per frame. A Viterbi pass with a switch penalty
   turns that into labels, then every dead stretch shorter than 3 s is removed (`possession.min_dead`).
2. **PathCRF** (trial 401) calls a team on every frame inside the BiGRU's alive stretches of at least 10 s (its window).
3. **Soft team vote** (`seq.soft_team`, since 2026-11-04): dead balls stay exactly as the BiGRU decodes them. On alive
   frames a Viterbi over H/A scores each team by the BiGRU's log P(team | alive), plus `BLEND_W = 4` for the team
   PathCRF calls. Where PathCRF has no call (stretches under 10 s), the BiGRU decides alone.

So the BiGRU owns: (a) dead vs alive everywhere, (b) where PathCRF's episodes start and end, (c) the team on the ~4%
of alive frames in short stretches, and (d) since the blend, a vote on the team everywhere else: on BAR-ATH a confident
BiGRU overrules PathCRF on 1.3% of the frames PathCRF covers.

Training data: the 7 public DFL (Bundesliga) matches with official labels, with the ball degraded to look like our
tracker's (`src/external.py`: same gap lengths, a dead ball that keeps moving, missing players). Never BAR-ATH labels.

## What QA says it is worth (`eda/NOTES.md`, "QA" section)

| DFL out of fold, mean of 7 | spellF1 / macroF1 |
|---|---|
| pipeline, hard PathCRF team (before 2026-11-04) | 0.589 / 0.887 |
| dead balls made perfect | 0.615 / 0.920 |
| only stoppage onsets fixed | 0.596 / 0.894 |
| team made perfect | 0.904 / 0.954 |

The dead-ball ceiling is about +0.03 spellF1 (and +0.03 macroF1, which is the tie-break). The short-stretch fallback
is wrong 30% of the time vs 9% for PathCRF, and causes 12% of all wrong-team frames. Those are the two targets.

The rows above use the hard team. The soft vote adds +0.014: on the `eda/team_blend.py` setup (PathCRF fold models in
official episodes, so a little optimistic) hard scores 0.588 / 0.890 and soft 0.602 / 0.891, better on 7/7 folds.

Tried by Eli on 2026-11-04 and not adopted: team-shape features (`SPACE=1`). They lift the BiGRU alone from 0.467 / 0.841
to 0.518 / 0.866, but not the pipeline (0.598 vs 0.602 with the soft vote), because PathCRF already carries that signal.

## How to run it

```
.venv/bin/python src/seq.py --cv        # leave-one-match-out on DFL, a few hours on CPU/MPS; prints spellF1 per switch
.venv/bin/python src/seq.py --train     # fit on all 7 -> src/seq_model.pt (replaces the submitted weights)
.venv/bin/python src/seq.py             # BAR-ATH -> evaluation/results/seq.csv + eda/out/seq_barath_logp.npy
.venv/bin/python src/seq.py --blend     # after src/pathcrf_run.py --trial N: soft team vote (--trial N, default 401) -> evaluation/results/pathcrf.csv
```
`--cv` writes `eda/out/seq_folds$TAG.npz` (`TAG` is an environment variable, empty by default). Set `TAG` for experiments
so the cached folds are not overwritten; Eli's folds on the current code are `TAG=_base`, with team shape `_space`.
`eda/qa.py` and `src/pathcrf_run.py --oof` only read the untagged `eda/out/seq_folds.npz`, so copy a tagged file there to
use them. The quickest whole-pipeline score is
`.venv/bin/python eda/team_blend.py seq:$TAG`: hard and soft team on the cached PathCRF fold outputs
(`eda/out/qa/pathcrf_20k_<match>.npy`, from `src/pathcrf_run.py --trial 20k --dfl <match> --save`), in seconds. Those use
official episodes; for episodes cut from the BiGRU's own dead balls, rerun the 7 folds with `--oof` (PathCRF venv, ~2 min
each; loop in the `eda/qa.py` docstring), then `.venv/bin/python eda/qa.py dfl` and read the `full` column.
Current numbers to beat: BiGRU 0.467 / 0.841 alone (`_base`), pipeline 0.602 / 0.891 with the soft vote.
In the Docker container (`docker/README.md`) the same commands work, with `python` in place of `.venv/bin/python`.

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
