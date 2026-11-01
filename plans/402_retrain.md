# Trial 402: PathCRF on DFL + PFF — Will

## Why 400 (DFL + PFF) lost to 401 (DFL only)

`meeting.md` says the World Cup run got 15 passes and the German-only run got 50, so the comparison was not fair. An
epoch is one pass over that run's training set, and 400's training set was about ten times bigger, so the epoch counts
hide what actually differed:

| trial | training matches | epochs | batch | gradient steps (relative to 401) | DFL share of training frames |
|---|---|---|---|---|---|
| 401 | 6 DFL | 50 | 32 | 1.0x | 100% |
| 400 | 6 DFL + 63 PFF | 15 | 64 | ~1.7x | ~9% |
| 200 (fold 0) | 5 DFL | 30 | 64 | — | 100% |
| 300 (fold 0 + PFF) | 5 DFL + 62 PFF | 15 | 64 | ~6.7x of 200 | ~7% |

(steps ∝ epochs × matches / batch, assuming PFF and DFL matches are about the same length; check this.)

So 400 took more optimiser steps than 401, not fewer. What it had less of was German data: each DFL match was seen 15
times instead of 50, and about 90% of its training frames carried PFF labels written by our hand rule. That rule,
applied to DFL's own events, reproduces the official labels at only spellF1 0.73-0.76, and it calls stretches dead that
look like live play. Our match's ground truth comes from corrected tracking, which DFL's official labels are the
closest thing to. The likeliest reading: 400 learned a blend of two possession definitions, mostly the wrong one.

That changes what 402 should be. "Same run, more epochs" mostly gives the PFF definition more time to win. The two
levers that address the cause:
1. **Teacher labels for PFF.** Already computed (`data_raw/external/pff_labels2_<gid>.npy`, from `src/label_transfer.py`:
   an HGB trained on DFL to reproduce the official labels, 0.752 / 0.949 vs official on DFL leave-one-out). Never wired
   into the PathCRF parquets.
2. **Oversample DFL** so the German definition is a real share of every epoch (each DFL training match repeated 4x is
   ~28% of frames instead of ~9%).

The evidence is also weaker than it reads:
- The ablation (300: 0.639 vs 200: 0.608) is one held-out match, and 300 had ~6.7x the gradient steps of 200. It shows
  that PFF plus far more steps did not hurt; it does not isolate PFF.
- J03WR9 is the validation match that picks the best checkpoint for both 400 and 401 (6 train + 1 val), so their J03WR9
  numbers (0.683, 0.707) are optimistic. Both were selected the same way, so comparing them is roughly fair, but a
  50-epoch run gets more chances at a lucky checkpoint than a 15-epoch run.
- The public window has 17 possessions (±0.05 noise). RULES.md says it is not a tuning set and the tech doc says nothing
  was tuned on it. 401 also wins on J03WR9, so the 401 choice stands without the window. Keep it that way for 402.

## Steps

1. **Teacher labels into the PFF parquets** (`src/pff.py`, function `pathcrf`).
   - Load `data_raw/external/pff_labels2_<gid>.npy` (H/A/D per frame). Assert its length equals `len(d["ball"])`: both
     should be on the 25-fps resampled frames, but verify.
   - Use it instead of `d["labels"]` for `lab`. That sets `ball_state`, `ball_owning_team_id` and the episode cuts.
   - PathCRF's targets (`poss_prev`, `poss_next`, `poss_team` in `~/pathcrf/dataset.py`) come from `player_id`, the
     carrier taken from PFF's on-ball events. Null `player_id` on frames where the carrier's team (`home_` / `away_`
     prefix) disagrees with the teacher's H/A, so the carrier never contradicts the label. Read how `dataset.py` derives
     `poss_team` before relying on this.
   - Behind a switch, `PFF_LABELS=rule|teacher` (default `teacher`), so 400's parquets stay reproducible.
   - Print per game: dead fraction (NOTES expects 0.30-0.41), number of episodes ≥ 10 s, rows with a carrier. Compare
     with the rule parquets.
2. **DFL oversampling** in PathCRF's `train.py` (record the change in `pod/pathcrf.patch`). Where the training file list
   is built from `SOURCE_CFG`, repeat the sportec training files `DFL_REPEAT` times (env var, default 1). Use 4.
3. **Pack and launch** (vast.ai, 1x RTX 4090; box setup in `pod/HANDOVER.md`).
   - `pod/vast.sh pack` ships only `pff_*.npz`. Add `data_raw/external/pff_labels2_*.npy`, and make sure `pff.py` on the
     box reads them when `setup` rebuilds the parquets.
   - `cd ~/pathcrf && TRIAL=402 SOURCES="sportec pff" DFL_REPEAT=4 BATCH=64 N_EPOCHS=30 nohup bash vast.sh full > full402.log 2>&1 &`
   - Same split as 400/401 (6 DFL train, J03WR9 val; 63 PFF train, 1 PFF val), so 402 is judged exactly like 401.
   - Cost: 400 ran 107 min/epoch. With DFL x4 expect ~120-130 min, so 30 epochs ≈ 60-65 h ≈ $30. Check the first
     epoch's time and recompute before walking away.
   - **Launch by Nov 6** so there is time to score it and hand Brandon a final trial.
4. **While it runs**: copy logs home; crash recovery is in `pod/HANDOVER.md`. The `logZ < gold` numerical check that
   stopped 401 at epoch 47 can recur; the best checkpoint saved before it is usable.
5. **Score it.**
   - `~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial 402 --dfl J03WR9`
   - `~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial 402 --alive <abs>/evaluation/results/seq.csv --out <abs>/evaluation/results/pathcrf_402.csv`
6. **Decide** (write the J03WR9 numbers into NOTES before looking at the public window):
   - adopt 402 if J03WR9 spellF1 ≥ 0.727 (401's 0.707 + 0.02) and macroF1 ≥ 0.949;
   - the public window can only veto: reject 402 if it is more than 0.05 below 0.660. Never adopt because of it;
   - otherwise 401 stays.
7. **If adopted**: upload `402.tgz` to the `pathcrf-weights` release; set the default trial in `src/pathcrf_run.py` and
   `pod/pathcrf_setup.sh` to 402; update TECH_DOC §4 and §6; tell Brandon. Destroy the box either way.

## If there is budget for a second run

The clean test of "does PFF help": the authors' fold-6 split (5 DFL train + PFF, val J03WQQ, J03WR9 never touched). That
gives a true held-out number to compare with the authors' checkpoint 140 (0.727, same split) and our 206 (0.655). Only
if 402 finishes early; it is another ~60 h.
