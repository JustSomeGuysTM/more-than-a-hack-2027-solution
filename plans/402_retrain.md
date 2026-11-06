# Trial 402: PathCRF on DFL + PFF — William Ngana (Will)

**Owner:** William Ngana (Will). Help: Elijah Cavan (Eli), for box access and the earlier runs.

## Quick start (steps 1-2 below are done; you start at step 3)

Status 2026-11-02: the PFF parquets now use the teacher labels by default (`PFF_LABELS=teacher` in `src/pff.py`), and
PathCRF's `train.py` repeats the DFL training matches `DFL_REPEAT` times (in `pod/pathcrf.patch`). Tested on the laptop:
3 PFF games built both ways and loaded through PathCRF's own dataset class with trial 401's settings.

| PFF game | rule: dead / episodes >= 10 s / carrier rows | teacher: dead / episodes / carrier rows |
|---|---|---|
| 10502 | 0.36 / 88 / 45,893 | 0.31 / 102 / 43,720 |
| 10503 | 0.34 / 116 / 44,793 | 0.30 / 108 / 42,995 |
| 10504 | 0.38 / 90 / 46,984 | 0.35 / 96 / 45,762 |

Training windows over those 3 games: 45,537 rule, 49,454 teacher. Carrier rows drop 2-5% because a carrier whose team
contradicts the teacher's label is now nulled.

**Branch:** the teacher labels and `DFL_REPEAT` live on `submission-fixes` (PR #7). Pack from that branch, or from
`main` once #7 is merged; packing from an older checkout silently trains on the old rule labels (check that `setup`
prints dead 0.30-0.41).

What you need: the vast.ai login (ask Eli), about $30-35 of credit, the SSH key `~/.ssh/vast` (or add your own public
key on vast.ai under Keys), and a laptop that has `~/mtah2027` and `~/pathcrf` (Eli's, or set yours up with
`bash pod/pathcrf_setup.sh` plus the data in the bucket).

1. **Rent** a 1x RTX 4090 on https://cloud.vast.ai with the "PyTorch" template and the disk slider at 50 GB. The
   Instances page shows the SSH command; note the port and host.
2. **Ship the code and data** (from the laptop that has the PFF caches; only public data goes to the box):
   ```
   bash ~/mtah2027/pod/vast.sh pack
   scp -i ~/.ssh/vast -P $PORT /tmp/pathcrf_box.tgz root@$HOST:~/
   ssh -i ~/.ssh/vast -p $PORT root@$HOST 'tar xzf pathcrf_box.tgz && find ~ -name "._*" -delete && bash ~/mtah2027/pod/vast.sh setup'
   ```
   `setup` ends with the number of PFF parquets (64) and "setup done". Spot-check one line of its output: dead should
   read 0.30-0.41.
3. **Launch** (it survives you logging out):
   ```
   ssh -i ~/.ssh/vast -p $PORT root@$HOST 'cd ~/pathcrf && TRIAL=402 SOURCES="sportec pff" DFL_REPEAT=4 BATCH=64 N_EPOCHS=30 nohup bash vast.sh full > full402.log 2>&1 &'
   ```
4. **Check it** after ~2.5 h, when epoch 1 is done; recompute the cost from the epoch time before walking away:
   ```
   ssh -i ~/.ssh/vast -p $PORT root@$HOST 'grep -c "^Epoch" ~/pathcrf/saved/402/log.txt; tail -3 ~/pathcrf/full402.log; nvidia-smi | head -15'
   ```
5. **When it ends** (`saved/402/model/state_dict_best_acc.pt` exists and the log says "Saved Model"), copy it home,
   then score and decide as in steps 5-6 below:
   ```
   scp -i ~/.ssh/vast -P $PORT -r root@$HOST:~/pathcrf/saved/402 ~/pathcrf/saved/
   scp -i ~/.ssh/vast -P $PORT root@$HOST:~/pathcrf/full402.log ~/mtah2027/pod/logs/
   ```
   Then destroy the instance on the vast.ai page (trash icon) so it stops billing.

Known box problems and fixes (all hit during trials 200-401) are in `pod/HANDOVER.md`: stale shells keeping
`pgrep -f train.py` true, macOS `._*` files, and the `logZ < gold` check that can stop a run near the end (the best
checkpoint saved before it is fine).

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

1. ~~**Teacher labels into the PFF parquets**~~ done 2026-11-02 (see Quick start) (`src/pff.py`, function `pathcrf`).
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
2. ~~**DFL oversampling**~~ done 2026-11-02 in PathCRF's `train.py` (record the change in `pod/pathcrf.patch`). Where the training file list
   is built from `SOURCE_CFG`, repeat the sportec training files `DFL_REPEAT` times (env var, default 1). Use 4.
3. **Pack and launch** (vast.ai, 1x RTX 4090; box setup in `pod/HANDOVER.md`).
   - `pod/vast.sh pack` now ships `pff_labels2_*.npy` too, and `setup` builds teacher-label parquets by default.
   - `cd ~/pathcrf && TRIAL=402 SOURCES="sportec pff" DFL_REPEAT=4 BATCH=64 N_EPOCHS=30 nohup bash vast.sh full > full402.log 2>&1 &`
   - Same split as 400/401 (6 DFL train, J03WR9 val; 63 PFF train, 1 PFF val), so 402 is judged exactly like 401.
   - Cost: 400 ran 107 min/epoch. With DFL x4 expect ~120-130 min, so 30 epochs ≈ 60-65 h ≈ $30. Check the first
     epoch's time and recompute before walking away.
   - **Launch by Nov 6** so there is time to score it and hand Brandon a final trial.
4. **While it runs**: copy logs home; crash recovery is in `pod/HANDOVER.md`. The `logZ < gold` numerical check that
   stopped 401 at epoch 47 can recur; the best checkpoint saved before it is usable.
5. **Score it.**
   - `~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial 402 --dfl J03WR9`
   - `~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial 402 --out evaluation/results/pathcrf_402.csv`, then
     `.venv/bin/python src/seq.py --blend --trial 402 --out evaluation/results/pathcrf_402.csv`: the soft team vote, as
     submitted since 2026-11-04. PathCRF's team calls go to `eda/out/pathcrf_barath_team_402.npy`, so 401's file
     (`_401`) and `evaluation/results/pathcrf.csv` stay as they are.
6. **Decide** (write the J03WR9 numbers into NOTES before looking at the public window):
   - adopt 402 if J03WR9 spellF1 ≥ 0.727 (401's 0.707 + 0.02) and macroF1 ≥ 0.949;
   - the public window can only veto: reject 402 if the score `seq.py --blend` prints is more than 0.05 below 0.660
     (401's, blended: 0.660 / 0.886). Never adopt because of it;
   - otherwise 401 stays.
7. **Refresh the QA videos for the new model** (whether or not it is adopted, so the team can compare; laptop, ~2 h):
   ```
   ~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial 402 --out evaluation/results/pathcrf_402.csv    # skip both if step 5 ran
   .venv/bin/python src/seq.py --blend --trial 402 --out evaluation/results/pathcrf_402.csv
   .venv/bin/python eda/qa.py barath --csv evaluation/results/pathcrf_402.csv --checks-only        # pipeline-bug checks
   .venv/bin/python eda/qa.py barath --csv evaluation/results/pathcrf_402.csv --video              # ~1 h, camera + 2D + labels
   aws s3 cp eda/out/qa/barath_changes.mp4 s3://more-than-a-hack-2027/qa_barath/barath_changes_402.mp4 --profile mtah
   ```
   The DFL clips (`eda/qa.py dfl`, `bash eda/qa_batch.sh`) come from the leave-one-out folds 200-206, not from 402, so
   they only change if new folds are trained. If 402 is adopted, also replace `s3://more-than-a-hack-2027/qa_barath/barath_changes.mp4`.
8. **If adopted**: upload `402.tgz` to the `pathcrf-weights` release; set the default trial to 402 in `src/pathcrf_run.py`,
   `pod/pathcrf_setup.sh` and `src/seq.py` (`TRIAL`); update TECH_DOC §4 and §6; tell Brandon. Destroy the box either way.

## If there is budget for a second run

The clean test of "does PFF help": the authors' fold-6 split (5 DFL train + PFF, val J03WQQ, J03WR9 never touched). That
gives a true held-out number to compare with the authors' checkpoint 140 (0.727, same split) and our 206 (0.655). Only
if 402 finishes early; it is another ~60 h.
