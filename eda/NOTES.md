# MTAH 2027 — team notes (Elijah, Oct 25)

## Reproduce PathCRF on BAR-ATH (any teammate, ~10 min + weights download)
```bash
pip install -r requirements.txt            # this repo's .venv, data/ unpacked as in README
gh auth login                              # needs access to the team repo (weights live in its GitHub release `pathcrf-weights`)
bash pod/pathcrf_setup.sh                  # clones+patches PathCRF into ~/pathcrf, venv, weights, BAR-ATH parquet, inference
```
Writes `evaluation/results/pathcrf.csv` (PathCRF home/away on alive frames, dead balls from our BiGRU; since 2026-11-04 the
team call is the soft blend `src/seq.py --blend`, PathCRF vote w=4, see "BiGRU and the PathCRF team blend") and prints the
public-window sanity score. `TRIAL=140` runs the authors' checkpoint (shipped in their repo) instead of our trial 401 (the submitted model).
Training itself is `pod/vast.sh` on a rented GPU (see its header); fold logs and CV numbers are in the PathCRF section below.

## TODO after the GPU run (for anyone on the team)
0. **PathCRF DFL + PFF at the long setting** (the one GPU experiment left; ~2 weeks of slack before Nov 17). The ablation
   said PFF helps at equal training length, the 15-epoch DFL+PFF final (400) lost to the 50-epoch DFL-only one (401).
   Run on the box (keep it alive or re-rent + `pod/vast.sh pack/setup`; the converter now applies all three PFF id fixes):
   `cd ~/pathcrf && TRIAL=402 SOURCES="sportec pff" N_EPOCHS=30 nohup bash vast.sh full > full402.log 2>&1 &`
   (batch 64, 30 epochs = 2x trial 400's steps, ~53 h, ~$25); or the authors' exact setting `BATCH=32 N_EPOCHS=50`
   (~5 days, ~$55). Judge like 400/401: `pathcrf_run.py --trial 402 --dfl J03WR9` (val match) and the BAR-ATH public
   window with the BiGRU dead balls; switch the candidate only if it beats 401 on both (0.707 / 0.954, and 0.660 / 0.886 after
   `src/seq.py --blend --trial 402`; steps in `plans/402_retrain.md`).
1. **Ball detector, second attempt** (biggest open idea; all pieces exist). The first attempt filled ATD gaps
   with YOLO detections and the dead class collapsed (see "Ball fusion result"). What to try instead:
   - Retrain the detector with more data: SoccerNet Tracking 2023 (public, in the bucket under
     `external/soccernet/SN-Tracking-2023/`) has ball boxes on broadcast frames; add them to the 2,269 ATD-projected tiles.
     Target: >5 m errors under 5% at conf 0.5 (now 24-42%).
   - Use detections as a *feature*, not a replacement: add "detector ball x/y/conf, frames since last detection"
     columns to `src/features.py` and let the HGB/BiGRU learn how much to trust them, with DFL training frames
     given synthetic detector noise (same error distribution as `eda/out/ball/detections.npz` vs ATD).
   - Only fill inside alive episodes (from `src/pathcrf_io.py`), never in dead stretches, and require the
     `ball_fuse.support()` 10/10 temporal consistency.
   - Judge it on DFL leave-one-match-out with the degraded ball, not on the public window.
   Entry points: `src/ball_detect.py` (training), `src/ball_fuse.py` (fusion + support filter),
   `src/v2.py --source tactical_fused`, weights `src/ball_yolov8n.pt` (not in git: bucket `derived/ball_yolov8n.pt`), detections in the bucket `derived/ball_detections.npz`.
2. **Dead-ball onsets** are still ~3 s late on our match after the carry fix (see "Dead-ball fix"): the restart end is now
   right, the stoppage start is not. Ideas: ball speed-jump / ball-off-pitch features with longer memory, or a dedicated
   alive/dead BiGRU with 60-s windows trained on augmented DFL.
3. **Ensemble** the HGB and BiGRU log-probabilities before Viterbi (cheap, DFL CV decides).
4. ~~Honest model selection~~ done: `degrade_ball` / `degrade_players` from measured ATD statistics (see "Make DFL look
   like BAR-ATH"); all CV numbers from 2026-10-28 on use it.
5. ~~PathCRF goalkeeper identity~~ done: PathCRF detects it itself, verified on our tracks (see below).
6. ~~Label transfer for PFF~~ done, negative for the frame models (see "Label transfer"); open: feed the teacher labels into
   the PFF parquets for PathCRF training.

## QA (DFL out of fold, PFF labels, BAR-ATH checks) (`eda/qa.py`, 2026-11-02)

**DFL out of fold.** PathCRF folds 200-206 (`src/pathcrf_run.py --dfl --save`) reproduce the fold table above exactly
(0.608 ... 0.655), and `eda/qa.py dfl`'s `pathcrf_gtdead` column equals them. BiGRU folds: `eda/out/seq_folds.npz`
(2026-10-28, realistic degradation, rescored 0.464 / 0.839). New: `--oof` runs PathCRF inside the BiGRU's out-of-fold
alive episodes, exactly as on BAR-ATH (the default `--dfl` run uses the official ball status for episodes, so PathCRF
never sees an officially dead frame). That is the honest `full` column; it costs only 0.001 vs official episodes.

| spellF1 / macroF1 (mean of 7) | |
|---|---|
| full pipeline (BiGRU dead balls + PathCRF team, BiGRU episodes) | **0.589 / 0.887** |
| same, PathCRF in the official episodes | 0.590 / 0.890 |
| BiGRU alone | 0.464 / 0.839 |
| PathCRF with GT dead balls (fold table) | 0.647 / 0.931 |
| full, stoppage onsets fixed from GT | 0.596 / 0.894 |
| full, dead balls exactly right | 0.615 / 0.920 |
| **full, team exactly right on alive frames** | **0.904 / 0.954** |

PathCRF ran on clean DFL tracking, so 0.589 is optimistic for BAR-ATH's degraded tracking. **Where the score goes: the
team call, not the dead balls.** Perfect dead balls add +0.026; perfect teams add +0.315.

Error types (full pipeline, 7 matches; seconds = frames actually wrong; a dead-ball error is typed once, as a dead-ball
error, never also as a team error):

| type | count | seconds | median s |
|---|---|---|---|
| missed_change (GT changes team, we stay on the old team > 2 s) | 494 | 819 | 1.4 |
| wrong_team_spell (> 50% of a >= 2 s GT possession on the other team) | 264 | 740 | 2.6 |
| spurious_change (we change team, GT does not within 2 s) | 323 | 492 | 1.4 |
| missed_stoppage | 54 | 339 | 4.8 |
| dead_onset_early | 108 | 333 | 2.1 |
| dead_onset_late | 66 | 269 | 2.5 |
| spurious_stoppage | 24 | 260 | 7.8 |
| restart_late | 70 | 256 | 2.3 |
| restart_early | 65 | 246 | 2.0 |

missed_change and wrong_team_spell overlap (a wrong spell starts with a missed change). Typing bugs fixed on the way:
a predicted dead run spanning two real stoppages was counted twice (late restart + up to 134 s "early onset"); team
errors used to swallow dead-ball frames; spurious changes and wrong spells were sized by the whole run, not the wrong
frames. `eda/qa.py selftest` covers each case.

What the clips show (`eda/qa.py clips --type <t> --n 5`, 2D pitch with the degraded ball the BiGRU saw):
- **Short alive stretches get no PathCRF.** Episodes under PathCRF's 10-s window are skipped, so the pipeline falls back
  on the BiGRU's team. That is 4% of alive frames, at 0.70 accuracy vs 0.91 where PathCRF runs, and 12% of all
  wrong-team frames. Typical clip: 6 s of play between two stoppages, a turnover after a throw-in, the BiGRU keeps the
  old team. Candidate fix (test on DFL CV with `--oof`): run PathCRF on short episodes padded to its window.
- **Midfield duels, ball hidden:** PathCRF switches team 3-6 s before the real turnover (spurious_change), or after it.
  88% of wrong-team frames are PathCRF's own; this is the model's ceiling on clean tracking.
- **Attacks into the box:** the BiGRU calls the ball dead ~4 s early while play goes on (dead_onset_early).

**PFF labels for trial 402 (`eda/qa.py pff`).** Teacher labels now go into the PathCRF parquets (`PFF_LABELS=teacher`,
default; carriers on the other team than the label are nulled) and load through PathCRF's dataset with 401's settings.

| | dead | stoppages | possessions | mean possession |
|---|---|---|---|---|
| PFF 10502 rule / teacher | 0.36 / 0.31 | 92 / 104 | 394 / 541 | 9.6 / 7.5 s |
| PFF 10503 rule / teacher | 0.34 / 0.30 | 118 / 108 | 370 / 470 | 10.6 / 8.8 s |
| PFF 3812 rule / teacher | 0.46 / 0.39 | 129 / 145 | 378 / 513 | 8.8 / 7.3 s |
| DFL official, 7 matches | 0.41-0.52 | 92-113 | 370-511 | 5.5-9.2 s |

Rule and teacher agree on 90-92% of frames and on the team on 95-97% of frames both call alive. The dead/alive
disagreements >= 2 s (45-77 per game, 4-6 min) are mostly rule-dead / teacher-alive. In all 4 clips checked, the teacher
was right: two throw-ins (ball back in play 3-5 s after going out; the rule waits 11-13 s for the next on-ball event),
a ball over the byline into a corner (teacher dead, rule alive 15 s), and the last 19 s around the final whistle
(teacher dead). Long stoppages agree. **Verdict: train 402 on the teacher labels.** Residual risk: the teacher's dead
fraction is ~0.1 below DFL's official one, so it may still call some restarts too early.

**BAR-ATH (`eda/qa.py barath`, submission = `pathcrf.csv`, byte-identical to `pathcrf_401.csv`).** Checks pass: no X on
graded frames, kick-off and half-time mostly D. Half 1: H 0.35 A 0.22 D 0.43, 242 possessions, 64 stoppages; half 2: H
0.31 A 0.22 D 0.47, 268 possessions, 70 stoppages (DFL range). 20-window change video: tracking overlays sit on the
players, the broadcast clock matches frame/25, no flipped half. Graded half 2 opens ~2 min before its kick-off (walk-out,
huddles; clock 45:06 at +125 s); the pipeline stays D until +117 s, which is right.

Overnight renders of DFL/PFF clips (public data only, never BAR-ATH) go to `s3://more-than-a-hack-2027/qa/` via
`eda/qa_batch.sh`.

## Dead-ball fix: DFL freezes the ball, BAR-ATH does not (`eda/dead_diag.py`, 2026-10-27)
Diagnostic on the 7 DFL folds (HGB v3): perfect dead frames would lift spellF1 only 0.557 -> 0.569, stoppage onsets/offsets
are within ±0.4 s, 3 of 711 stoppages missed. So on DFL the dead ball is solved. On BAR-ATH's public window it is not:
D starts 1.4-2 s late and ends 2-6 s early (545 D frames predicted of 1469; 2 of 4 stoppages missed by v2).
Cause: in DFL the ball is **frozen while dead** (speed exactly 0 on 100% of dead frames, always seen, ~5 m from everyone);
the BAR-ATH tracker keeps reporting a **moving** ball during stoppages (median 2.7 m/s, 35% within 1.5 m of a player who
carries it to the restart, p75 6.5 m away; only 1% of dead frames have speed < 0.5 m/s). The models learned "still ball =
dead", a cue our match never shows.
Fix (`src/external.py: dead_ball_carry`, on by default in `src/v2.dfl_game`, so HGB and BiGRU both get it): per DFL dead run,
60% attach the ball to the nearest player at the stoppage with a random-walk offset, 40% let it drift from where it stopped.
Augmented DFL dead frames: speed p50 1.6 m/s, nearest player 3.0 m. A first version (always carried, fixed offset) fired on
slow alive play (25 false stoppages in the public window) and is kept only as a note.
Plus `src/possession.min_dead`: D runs < 3 s take the preceding label (T chosen on DFL CV: best or tied for every variant,
2.7% of DFL stoppages are shorter). `switch_d` grid on DFL is flat (0.52-0.54), so the decoder penalty is not the lever.

| | DFL CV (held-out also augmented) spellF1 / macroF1 | public 5 min (sanity) spellF1 / macroF1 / spells |
|---|---|---|
| HGB v3, frozen DFL ball (before) | 0.557 / 0.909 | 0.391 / 0.686 / 27 |
| HGB v3 + carry + min_dead 3 s | 0.532 / 0.893 | **0.537 / 0.834 / 18** |
| PathCRF ckpt + v2 dead (before) | – | 0.615 / 0.716 / 24 |
| **PathCRF ckpt + new HGB dead** (`evaluation/results/pathcrf.csv`) | – | **0.622 / 0.853 / 28** |

The DFL number drops because the augmented task is harder (and closer to ours); it is the honest one from now on. Public window:
dead-frame F1 0.92 -> 0.95, D onsets still ~3 s late (the last open dead-ball issue), 3 short false stoppages remain.
BiGRU with the augmentation (DFL CV, held-out augmented): **0.582 / 0.907** at switch 3 (6: 0.579, 10: 0.576), vs HGB 0.532 / 0.893
on the same setup; `src/seq_model.pt` retrained (2026-10-28). Public window (sanity): BiGRU alone 0.644 / 0.760, 19 spells,
but its dead balls are weaker than the HGB's (758 D frames of 1469, onset +3.8 s, offset -2.1 s, dead-frame F1 0.925 vs 0.954),
and as PathCRF's dead source it scores 0.545 / 0.763 vs 0.622 / 0.853 with the HGB. **HGB stays the dead-ball source**;
the BiGRU is the standalone fallback candidate. Tested 2026-10-28 (`seq.py --cv` now caches fold log-probs in `eda/out/seq_folds.npz` and
scores the combinations against `dead_folds_carry2.npz`): BiGRU alone 0.585 / 0.904; BiGRU H/A + HGB D (`seq.combine`)
0.575 / 0.901; mean of the two log-prob sets 0.584 / 0.907 (switch 6). No gain: the BiGRU's own dead balls are as good as
the HGB's on DFL, and the public-window gap between them is window noise. Negative result; the BiGRU stays standalone.
Old results kept: `evaluation/results/pathcrf_v2dead.csv` (PathCRF + v2 dead, 0.615 / 0.716).

### Ball detector re-test after the dead-ball fix (2026-10-27)
The fusion failed before because fills on dead frames turned D into H/A; with `dead_ball_carry` the model no longer reads a
moving ball as "alive", so the same detections (`--source tactical_fused`, conf 0.25) now help on the public window (sanity):
HGB 0.537 / 0.834 / 18 spells -> **0.544 / 0.865 / 19**, dead-frame F1 0.954 -> 0.967; PathCRF (stitched on the fused ball)
+ fused dead balls **0.664 / 0.880 / 27** vs 0.622 / 0.853 unfused (`evaluation/results/pathcrf_fused.csv`). Retraining
the HGB at visibility 0.78 fragments again (0.417, 31 spells): keep 0.57. Still no honest validation of the fused ball
(DFL has no detector noise); within the window's ±0.05 noise band. Next if pursued: SoccerNet Tracking ball boxes to
retrain the detector, and a detector-noise simulator on DFL so the choice can be made on CV.

### Detector-noise simulator on DFL: the fused ball is neutral (2026-10-27)
`src/external.simulate_detector` (FILLS=1 for `src/v2.py` / `eda/dead_diag.py`) reproduces our detector on a degraded DFL ball:
fills ~14% of hidden frames in runs of p50 4 / p90 14 frames, errors p50 0.8 m, 6.5% > 5 m (targets from the real detections:
18%, 3 / 20 frames, 0.77 m, 11%). HGB leave-one-out, augmented dead ball, min_dead 3 s:

| DFL ball | train | spellF1 / macroF1 |
|---|---|---|
| degraded only (= current pipeline) | no fills | 0.532 / 0.893 |
| degraded + simulated fills | with fills | 0.538 / 0.894 |
| degraded + simulated fills | no fills (what gave 0.664 on the public window) | 0.530 / 0.889 |

+0.006 at best, inside fold noise. The public-window gain (0.622 -> 0.664) is not supported by the honest check, and the
fused pipeline costs a 2.5-h detector pass for anyone reproducing the CSV. **Decision: the detector stays out of the
submission pipeline**; the simulator stays so a better detector (SoccerNet-trained) can be judged the same way.

### Ball-detector ceiling check (2026-11-03, plans/soccernet_ball_detector.md step 1)
Same simulator with a better detector (`DET=fill,p_gross,sigma FILLS=1 TAG=_detN .venv/bin/python eda/dead_diag.py`), on the
**current** degradation (real ATD gap structure + player dropout; the 2026-10-27 table above used the older, gentler one).
HGB v3 leave-one-out, train and test with the same fills. "Pipeline" = that model's dead balls + PathCRF folds 200-206 for
the team (official episodes, i.e. `full_gtep` in the QA table; the BiGRU row reproduces its 0.590).

| detector (share of hidden frames filled, gross > 5 m, typical error) | HGB alone | pipeline | dead/alive acc |
|---|---|---|---|
| none (current) | 0.438 / 0.828 | 0.580 / 0.884 | 0.947 |
| ours (`DET` unset: ~16%, 11%, 0.65 m) | 0.505 / 0.875 | 0.597 / 0.901 | 0.966 |
| good (`DET=0.65,0.02,0.3`: ~50%, 2%, 0.3 m) | 0.563 / 0.900 | 0.610 / 0.910 | 0.977 |
| great (`DET=1.5,0.02,0.3`: ~80%, 2%, 0.3 m) | 0.567 / 0.904 | 0.615 / 0.913 | 0.980 |
| perfect (`DET=5,0,0`: 100%, 0, 0) | 0.604 / 0.914 | 0.627 / 0.920 | 0.987 |
| BiGRU, no fills (submission) | | 0.590 / 0.890 | 0.953 |

- A better ball is worth +0.02 (good) to +0.025 (great) on the pipeline over today's BiGRU, +0.037 at the ceiling. That clears
  the plan's ~0.02 bar, just: SoccerNet training is worth doing for Phase 1, not only the innovation round.
- Under the current degradation **even our existing detector helps** (+0.017 over the same HGB without fills, +0.007 over the
  BiGRU). The "neutral, out of the pipeline" decision above was made on the old degradation. Not reopened here: the gain
  is small, it costs the 2.5-h detector pass, and the BiGRU would need retraining with fills (`src/seq.py` reads `src.v2.FILLS`).
- Caveats: simulated detector (errors independent of game state; a real one fails in crowds and at the far side, exactly
  near stoppages); PathCRF in official episodes is optimistic; HGB, not the BiGRU, made the dead balls.
- Leak to avoid: an earlier version left dead frames unfilled ("the detector is off when the ball is dead"); a still-missing
  ball then means dead, and the scores jumped. Fills now land on dead frames too.

### SoccerNet ball data, day one (2026-11-03, `src/soccernet.py`)
- Format confirmed: MOT `gt/gt.txt` (frame, track, x, y, w, h, ...), ball track named `ball;1` in `gameinfo.ini`. Train
  split: 57 clips x 750 frames, 40,931 ball boxes, ball labelled on 94% of frames (clip p10 81%).
- **Domain gap:** SoccerNet ball box sqrt(w*h) p5/50/95 = 8 / 14 / 23 px (zoomed broadcast). Ours on the tactical feed,
  0.22 m through the homography at our detector's confident hits: 3 / 4.2 / 7 px. A ~3.3x scale difference.
- **Zero-shot baseline** (stock COCO `yolov8n.pt`, class 32, top-1 per frame, 5 clips x 50 frames, hit = within max(8 px, box)):

| frames | recall | wrong or no-ball top-1 |
|---|---|---|
| SoccerNet as is | 0.58 | 44% |
| shrunk to our ball size (x0.3) | 0.18 | 10% |
| shrunk x0.3, then upsampled x2.2 by YOLO | 0.53 | 33% |

- The ball survives the shrink; the network's stride loses it. So: **run our detector on upsampled tiles** (2-3x; about
  6x the tiles, so a few hours for the full match on MPS) and train on SoccerNet frames shrunk to our size then upsampled
  the same way. Upsampling is cheap to test with our existing weights first.
- Caveat: a shrunk broadcast frame is sharper than our wide camera (different lens, compression, motion blur); the real
  test is on our feed against the ATD.

### Upsampled tiles on our feed: negative (2026-11-03)
`src/ball_detect.py --infer --up 2` (frame upsampled 2x, 8x4 tiles of 640 instead of 4x2), every 5th frame of BAR-ATH
(2 h 20 min on MPS); stock COCO yolov8n at x1 and x2 on every 50th frame. Scored with `--check <file> --step 50` against the
ATD ball (3,987 frames with a homography, ATD ball on 57%; includes warm-up and stoppages, so absolute numbers are worse
than the 2026-10-27 table, which used alive frames only):

| top-1 detector, conf 0.25 | detects on ATD-ball frames | median error vs ATD | > 5 m off | detects on ATD-missing frames |
|---|---|---|---|---|
| ours, native (`detections.npz`) | 89% | 1.3 m | 28% | 38% |
| ours, x2 | 99% | 16.5 m | 69% | 95% |
| COCO, native | 68% | 13.0 m | 58% | 42% |
| COCO, x2 | 93% | 14.5 m | 59% | 77% |

- Upsampling **hurts** our detector. Pixel positions map back correctly (identical to native where both find the ball);
  with 4x the tiles, the top-1 box is more often a confident false positive: a player's head, a line, a spare ball in
  the warm-up. The model was trained on native tiles (ball ~4 px) and never saw an 8 px ball or that many negatives.
- Stock COCO is not a usable baseline on our feed at either scale (median error 13-15 m). The SoccerNet result (shrink then
  upsample recovers recall) does not transfer as-is: our frames are blurrier and the false-positive load is far higher.
- So the next detector has to be **trained** at the inference scale with many hard negatives: SoccerNet frames shrunk to
  our ball size (x0.3) then upsampled x2, ball-free tiles from our feed (allowed: no labels involved), and judged on
  top-1 precision, not recall. Plus the temporal layer (plan step 4), since per-frame top-1 is where it fails.
- Detections stay local (`eda/out/ball/detections_*`); they are derived from BAR-ATH.

### Detector trained at the x2 scale: better than x2 with old weights, still worse than native (2026-11-03)
`src/soccernet.py --build` (SoccerNet frames shrunk so the ball has our native size 3-7 px, blurred sigma 0.3-0.8, upsampled
x2 like inference; one ball + one ball-free tile per 5th frame; 47 train / 10 held-out clips) + our ATD tiles from
`data_raw/ball` upsampled x2: 8,671 ball / 7,857 empty train tiles. `--train`: YOLOv8n from COCO, 25 epochs, 3.4 h on MPS.
Held-out tiles (SoccerNet + ours): P 0.59, R 0.51, mAP50 0.43. Weights `src/ball_yolov8n_up2.pt` (gitignored, bucket
`derived/ball_yolov8n_up2.pt`: it saw our match's tiles). On our feed, every 50th frame **offset 25** (never a training tile;
`--check ... --step 50 --offset 25`):

| top-1, conf 0.25 | detects on ATD-ball frames | median error vs ATD | > 5 m off | detects on ATD-missing frames |
|---|---|---|---|---|
| native, current weights | 90% | 1.4 m | 29% | 38% |
| x2, current weights | 100% | 17.0 m | 70% | 96% |
| **x2, new weights** | 81% | 2.4 m | 42% | 42% |

- Training at the scale fixes most of the x2 collapse, but it does not beat the native detector, so it does not
  replace it. Gross errors (42% > 5 m) are what the dead-ball model cannot afford (ceiling check: "good" means 2%).
- Caveat in the native model's favour: it was trained only on ATD-projected labels, so it shares the ATD's choices (e.g.
  which ball in the warm-up); agreement with the ATD is not the same as being right.
- Not tried, in order of promise: (1) SoccerNet at **native** scale added to the current native set (no upsampling, same
  tiles at inference); (2) the temporal layer on top-k candidates (plan step 4): per-frame top-1 is where every
  variant fails, and a smooth-track filter is what turned 42% gross into 12% for `ball_fuse.support`.

### Temporal layer on top-k candidates (2026-11-03, `src/ball_track.py`)
Viterbi over each frame's top-5 detections plus a "no ball" state: smooth motion is cheap (Gaussian on the step, hard cap
1.6 m/frame = 40 m/s), switching the track on/off costs 4, tracks shorter than 5 frames are dropped. Self-test on a
synthetic track with confident one-frame false positives. Candidates: current native weights, `--topk 5 --window 100`
(dense stretches f % 500 < 100, ~20% of the match). Scored on frames f % 50 == 25 (never training tiles).

**Score on graded play only** (`--graded`; ball_fuse's graded ranges). Outside it (warm-up, half-time) several balls are on
the pitch and the ATD follows one: 77% of tracked frames there are "> 5 m off", which dominated the earlier tables.

| graded play, 624 frames | detects on ATD-ball frames | median error | > 5 m off | detects on ATD-missing frames |
|---|---|---|---|---|
| top-1, conf 0.25 | 87% | 0.9 m | 15% | 49% |
| top-1, conf 0.6 | 48% | 0.7 m | 11% | 14% |
| **tracked** | 87% | 0.9 m | 12% | 41% |
| x2 weights, top-1 conf 0.25 (every 50th +25, 3,118 frames) | 77% | 1.4 m | 30% | 49% |

- The tracker keeps conf-0.25 coverage at about conf-0.6 precision. Remaining gross errors: in 81% of them no top-5 candidate is
  within 2 m of the ATD, so the detector never saw that ball (or the ATD is wrong); smoothing cannot fix those.
- Parameter sweep (on / none / min_len / sigma, chosen on even windows, reported on odd): the pick did not transfer
  (10% -> 14% gross); defaults kept. Lower gross (6%) costs coverage (62-68%).
- **Worth on DFL:** that profile in the simulator (`DET=0.5,0.12,0.75`: ~40% of hidden frames, 12% gross, 0.9 m) gives
  pipeline **0.603 / 0.905** vs 0.590 / 0.890 with today's BiGRU (+0.013) and 0.597 for the untracked detector. Real, small; the
  "good detector" (0.610) needs gross errors near 2%. Not adopted: needs a full-match top-k pass (~2.5 h + decode) and a
  BiGRU retrain with fills, for +0.01 inside fold noise.

### SoccerNet at native scale: the model barely fires on our feed (2026-11-04)
`src/soccernet.py --build --up 1` / `--train --up 1`: SoccerNet frames shrunk to our ball size (no upsampling) + our ATD tiles,
YOLOv8n from COCO, 25 epochs (3.5 h). Held-out tiles: P 0.60, R 0.49, mAP50 0.43. Weights bucket `derived/ball_yolov8n_sn.pt`.
On our feed (top-5, dense windows, graded play): detects on only **10%** of ATD-ball frames at conf 0.25 (current weights 87%),
3% at 0.4; tracked 5%. When it fires it is right (0.8 m median), but it is far too unsure on our frames.
Likely cause (untested): label boxes. This set labels the ~4 px ball with a 6 px box (SoccerNet boxes scaled the same way);
the current weights were trained with 20 px boxes around the same ball, which gives the stride-8 head context. Next
if anyone continues: rebuild with 20 px boxes at native scale, or fine-tune the current weights on SoccerNet instead of
training from COCO.

### Colour-order bug in `ball_detect.infer` (fixed 2026-11-04)
ffmpeg decodes rgb24 and `infer` passed those arrays straight to ultralytics, which treats arrays as BGR (and reads the training
jpgs as BGR). So every detection pass so far ran colour-swapped. On SoccerNet (clip SNMOT-062, 30 frames at our ball size, our
weights) the ball is among the candidates on 11/30 frames swapped vs **21/30** in the right order. Fixed with one line; the old
top-5 files are kept as `eda/out/ball/*_rgb.npz` so the tables above stay reproducible. Every detector number above (87% / 0.9 m /
12% gross) was measured with the swap: reruns with the fix are in progress. The submission never reads detector output
(`v2.py --source tactical` default), so it is unaffected.

### Learned candidate scorer (`src/ball_scorer.py`, 2026-11-04)
The "learned version" of plan step 4. Our weights (now in BGR) on every frame of all 57 SoccerNet clips shrunk to our ball size
(4.2 px), CPU, top-5 merged as `infer` (26,934 candidates). Per frame and slot: present, conf, and for frames t-2, t-1, t+1, t+2 the
distance to the nearest candidate (capped 40 px) and its conf. A 2-layer BiGRU (hidden 48) picks a slot or "none"; label = slot
within max(3 px, ball size) of the human box. 47 clips train, last 10 held out (as `soccernet.VAL_CLIPS`), 60 epochs of random
8 s windows. Pixel space on purpose, so it transfers to our feed (columns u, v of the detection files; both 25 fps).

| held-out SoccerNet, 2,347 frames with the ball among the candidates | picked right | wrong picks |
|---|---|---|
| top-1 at conf 0.25 | 71% | 617 |
| **GRU scorer (p >= 0.5)** | **75%** | **456** |

Modest, and noisy across epochs (62-75%). Not yet run on our feed: needs the BGR top-5 rerun (`--ours <file>` scores it next to
the Viterbi tracker on the same frames). Ideas for whoever picks it up: add player context (distance to the nearest player), more
neighbour frames, train on candidates from whichever detector wins.

### Team shape: convex hull and Voronoi share (`src/features.py`, `SPACE=1`, 2026-11-04)
Idea from Elijah's Big Data Bowl 2022 work (penalised expected arrival time / space control). Per team: convex-hull area,
share of the pitch it controls (Voronoi = nearest player, on a 21x14 grid), share of the grid within 15 m of the ball; plus ±0.5 s
and ±2 s rolling means. 12 s per match. Behind a flag so the default features (and the BiGRU) are unchanged.
Metrica match 1 (degraded ball), AUC separating H from A possession frames: **hull difference 0.775**, Voronoi 0.664, near-ball
share 0.621; current best team feature `ddiff` 0.685. The team in possession spreads out, and the hull needs no ball (missing
on 43% of our frames). DFL leave-one-out (HGB, `SPACE=1 TAG=_space eda/dead_diag.py`, same code and degradation as `_det0`):

| DFL 7 folds, HGB | spellF1 / macroF1, own dead balls | with GT dead balls (oracle D) | false D runs |
|---|---|---|---|
| `_det0` baseline | 0.438 / 0.828 | 0.474 / 0.869 | 56 |
| **+ team shape (`SPACE=1`)** | **0.486 / 0.856** | **0.528 / 0.897** | 52 |

+0.048 spellF1. Most of it is the team call (the oracle-D row gains as much); dead-ball detection is about unchanged (missed
stoppages 36 -> 39, false runs 56 -> 52). Biggest single feature gain since the since/until features. Next: BiGRU retrain with
`SPACE=1` (Brandon), and a PathCRF + HGB/BiGRU team blend, since PathCRF supplies the submission's team call.

**BiGRU and the PathCRF team blend** (`eda/team_blend.py`, 2026-11-04). The submission is BiGRU dead balls + PathCRF team, so
these rows decide. BiGRU folds rerun on current code (`TAG=_base`; the Oct 28 cache gives the same 0.590 hard).
*Hard* = today: PathCRF's team wherever it covers the frame. *Soft w* = dead balls exactly as the BiGRU decodes them; on its
alive frames a Viterbi over H/A only, with the BiGRU's team log-prob plus a bonus w for PathCRF's team, so a confident BiGRU can
overrule PathCRF. (A first soft version also let the bonus raise "alive": PathCRF's DFL outputs only cover truly alive frames,
so that leaked the true dead balls and scored 0.621, rising with w. Discarded.)

| DFL 7 folds, spellF1 / macroF1 | BiGRU alone | hard (today) | soft w=2 | soft w=4 |
|---|---|---|---|---|
| BiGRU | 0.467 / 0.841 | 0.588 / 0.890 | 0.597 / 0.890 | **0.602 / 0.891** |
| BiGRU + team shape (`SPACE=1`) | 0.518 / 0.866 | 0.585 / 0.888 | 0.601 / 0.890 | 0.598 / 0.889 |

Per fold vs today (hard, plain BiGRU): **soft w=4, plain BiGRU: +0.014, better on 7/7 folds** (+0.007 to +0.018). Team shape
lifts the BiGRU alone by +0.051, but with PathCRF on team it adds nothing under hard (-0.003, 2/7) and about +0.01 under soft:
PathCRF already captures the shape signal, and the features don't improve the dead-ball calls. w picked on the same folds (only
2 and 4 tried; both positive). Adopting soft w=4 is a decode change only (BiGRU log-probs + PathCRF labels, no retraining).
**Adopted 2026-11-04** (`src/seq.py --blend`, `BLEND_W = 4`, last step of `pod/pathcrf_setup.sh`; same `soft_team` function as
the CV). Before switching, `seq.csv` and the hard `pathcrf.csv` both reproduced byte-identical. BAR-ATH: 1,147 of 199,319 frames
change (team only, dead balls identical; 1.3% of PathCRF-covered alive frames); public window (sanity only) 0.660 / 0.886 vs
0.660 / 0.884 hard. `validate_submission.py`: valid.

### Detector reruns with the colour fix (2026-11-04)
Top-5, dense windows, graded play (624 frames), vs the ATD; old numbers (colour-swapped) in brackets.

| detector | top-1 conf 0.25: detects / >5 m off | tracked (Viterbi): detects / >5 m off | detects on ATD-missing (tracked) |
|---|---|---|---|
| current weights, BGR fix | 89% / 13% (87% / 15%) | **88% / 10%** (87% / 12%) | 46% (41%) |
| SoccerNet 20 px boxes, fine-tuned from ours (`sn20ft`) | 91% / 16% | 89% / 12% | 52% |
| GRU scorer (`ball_scorer.py`) on current BGR candidates, p >= 0.5 | | 80% / 11% (p 0.3: 85% / 13%) | 46% |

- Colour fix: a small real gain (gross 12% -> 10% tracked). The fine-tune fires on more frames the ATD misses (52% vs 46%), which
  can't be checked against the ATD, but on checkable frames it is no more accurate: not adopted. Weights `src/ball_yolov8n_sn20ft.pt`.
- The SoccerNet-trained scorer does not beat the hand-set Viterbi on our feed at any threshold: negative transfer. Parked.

Where it would land: the BiGRU dead-ball model reads these features directly; the team call comes from PathCRF, so for team
it would need a blend, or use where PathCRF loses players.

### BiGRU on BAR-ATH in the Docker container (2026-11-05)
`python src/seq.py` in the pipeline container (`docker/README.md`: CPU, torch 2.5.1, the organisers' numpy 1.26.4 pins) with
the committed `src/seq_model.pt`: public window (sanity only) **0.625 / 0.879, 15 spells, alive/dead F1 0.968**, the same as the
2026-10-28 numbers for this model. So the container reproduces the BiGRU's dead balls at least to the score; the byte-level
check against the submitted `seq.csv` is still checklist step 4. The run also writes `eda/out/seq_barath_logp.npy` for
`--blend`; the blend itself needs PathCRF's team from `pod/pathcrf_setup.sh` (run `gh auth login` in the container first).
The run took 14 min on the container's CPU. Team shape is not retrained in (see "Team shape" above: no pipeline gain).

PathCRF in the container, same day: `src/pathcrf_io.py` (5 s), then `src/pathcrf_run.py --trial 140` (authors' checkpoint,
CPU inference 128 s) on those BiGRU dead balls: public window **0.673 / 0.881, 27 spells**, vs **0.605 / 0.875, 26 spells**
recorded on 2026-10-29. `src/pathcrf_io.py` is unchanged since then and the 2026-11-02 patch change only touches training, so
the gap points at the environment (the container pins numpy 1.26.4 / pandas 2.2.3 / torch 2.5.1; the laptop's PathCRF venv
was installed unpinned) or at the parquet being rebuilt here. Not resolved. Until the container installs the laptop's frozen
PathCRF versions (`plans/submission_checklist.md` step 2), container PathCRF numbers are not reproductions. The soft vote on
top changes 1.1% of PathCRF's calls and scores the same, 0.673 / 0.881.
Blend inputs are now per trial: `src/pathcrf_run.py` writes `eda/out/pathcrf_barath_team_<trial>.npy`, and
`src/seq.py --blend --trial N --out F` reads it, so scoring 402 no longer replaces 401's team calls (`plans/402_retrain.md` step 5).

## PathCRF own training on vast.ai (RTX 4090, `pod/vast.sh`, queue in `~/pathcrf/queue.sh` on the box; 2026-10-28)
Fold 0 (trial 200, DFL-only, held-out J03WMX, 30 epochs x 8.1 min, best val acc at epoch ~12) scored with
`src/pathcrf_run.py --trial 200 --dfl J03WMX` (GT dead frames substituted, PathCRF team on covered alive frames):

| J03WMX held out, GT dead for all | spellF1 | macroF1 | spells (gt 511) |
|---|---|---|---|
| HGB v3 (augmented, fold 0) | 0.509 | 0.916 | 312 |
| BiGRU (augmented, fold 0) | 0.568 | 0.925 | 325 |
| **PathCRF ours, trial 200** | **0.608** | **0.933** | **419** |
| authors' 140 (J03WMX was in its training set) | 0.706 | 0.955 | 399 |

Authors' checkpoint 140 is itself a leave-one-out fold (`--lomocv --cv_fold_idx 6`: held-out J03WR9, val J03WQQ, 5 training
matches, batch 32, 50 epochs, seed 100; everything else identical to ours), so it *is* documentable: public DFL only.
Its own held-out score with this scorer: 0.727 / 0.949 (335 spells, gt 370). Direct comparison comes from our trial 206.
Fold 1 (trial 201, held-out J03WN1, 2026-10-28 13:50 UTC): **0.579 / 0.905**, 427 spells (gt 500), alive frame acc 0.858;
authors' 140 on the same match (in its training set) 0.717 / 0.939. HGB (realistic degradation, GT dead) on the same matches for scale:

| fold | held out | PathCRF ours (GT dead) | HGB real-degradation (GT dead) |
|---|---|---|---|
| 0 | J03WMX | 0.608 / 0.933 | 0.451 / 0.870 |
| 1 | J03WN1 | 0.579 / 0.905 | 0.442 / 0.853 |
| 2 | J03WOH | 0.691 / 0.940 | 0.485 / 0.869 |
| 3 | J03WOY | 0.648 / 0.926 | 0.466 / 0.865 |
| 4 | J03WPY | 0.673 / 0.933 | 0.469 / 0.865 |
| 5 | J03WQQ | 0.673 / 0.938 | 0.528 / 0.874 |
| 6 | J03WR9 | 0.655 / 0.943 (authors' 140, same held-out: 0.727 / 0.949) | 0.480 / 0.884 |
| **mean, 7 folds** | | **0.647 / 0.931** | 0.474 / 0.868 |

Run 1 complete 2026-10-29 07:40 UTC. Our 30-epoch / batch-64 folds trail the authors' 50-epoch / batch-32 fold by 0.07 on
the one directly comparable match. Fold-6 model on BAR-ATH (sanity): 0.500 / 0.873, 31 spells -- fragments our match where
the authors' checkpoint gives 26 (0.605 / 0.875). Both PFF stages crashed at start (`KeyError: 'away_None_x'`: 1,253 OTB
rows across the 64 parquets carried a jerseyNum of None; fixed in place on the box and in `src/pff.py`). New queue
(`~/pathcrf/queue2.sh`): **trial 401** = DFL-only final with the authors' settings (6 train + 1 val, batch 32, 50 epochs,
~8 h) -> **trial 300** PFF ablation fold 0 (15 epochs, ~27 h) -> **trial 400** DFL+PFF final (15 epochs, ~27 h).
Candidate weights: 401 unless the ablation says PFF helps.
**Trial 401 done 2026-10-29 15:13 UTC** (DFL-only, authors' settings; training raised PathCRF's `logZ < gold` numerical
check at epoch 47 of 50, the best-accuracy checkpoint from before that is what we use). On BAR-ATH with the BiGRU's dead
balls (sanity): **0.660 / 0.884, 23 spells** -- best so far (authors' checkpoint 0.605 / 0.875, our fold-6 model 0.500 /
0.873). `evaluation/results/pathcrf.csv` is now this; `pathcrf_run.py --trial` defaults to 401.
Trial 300 crashed a second time at start (`KeyError: 'home_1_x'`: 290 OTB rows named jerseys that never appear in the
tracking columns of their match; nulled in place on the box). `queue3.sh`: 300 (~27 h, ~Fri 23:00 UTC) -> 400 (~Sun 02:00 UTC).
**Trial 300 done 2026-10-30 ~19:30 UTC** (PFF ablation: fold 0, 5 DFL + 62 PFF train, 15 epochs x 101 min). Held-out
J03WMX: **0.639 / 0.940** vs DFL-only fold 0 (trial 200, 30 epochs) 0.608 / 0.933 -> PFF helps PathCRF (+0.03 on one
match; note 300 also took ~5x the gradient steps). BAR-ATH sanity: 0.594 / 0.876, 18 spells (`pathcrf_300.csv`).
(Correction: 300 was still in its 15th epoch at that check; the `home_13_x` KeyError in `full400.log` was the Oct 29
crash.) Before 400 starts, a third data fix applied preemptively: PathCRF drops a player's columns inside an episode when
they are all-NaN there, so an OTB id must be tracked on its own frame; 457 such rows nulled on the box and the rule added
to `src/pff.py` (`tracked`). 400 starts automatically after 300 (~21:15 UTC), done ~Sun 2026-11-01 00:00 UTC.
**GPU run complete 2026-10-31 23:52 UTC** (~4.5 days, ~$42). Trial 400 (DFL + 63 PFF final, 15 epochs x 107 min):
BAR-ATH sanity 0.556 / 0.855, 24 spells; DFL val match J03WR9 0.683 / 0.947 (293 spells, under-segments).
Trial 401 (DFL-only, authors' settings, 50 epochs): 0.660 / 0.884 and 0.707 / 0.954 on the same two. **Decision: submit
401.** PFF helped at equal short training (ablation) but the long DFL-only run wins on both checks; a DFL+PFF run at the
authors' length would take ~4 days more and is not planned. All weights (200-206, 300, 400, 401) are on the laptop and in
the GitHub release `pathcrf-weights`; `pod/logs/` has every training log. Box destroyed after this.

| final models | DFL val J03WR9 (GT dead) | BAR-ATH public (BiGRU dead, sanity) |
|---|---|---|
| authors' checkpoint 140 (their LOMO fold, J03WR9 held out) | 0.727 / 0.949 | 0.605 / 0.875, 26 spells |
| **ours 401, DFL only, 50 epochs** | **0.707 / 0.954** | **0.660 / 0.884, 23 spells** |
| ours 400, DFL + PFF, 15 epochs | 0.683 / 0.947 | 0.556 / 0.855, 24 spells |
Incident: the queue stalled 1 h after fold 0 because two stale watcher shells whose command lines contained `train.py`
kept `pgrep -f train.py` true; killed, folds 1-6 running. Fold logs: `pod/logs/`.

## Make DFL look like BAR-ATH: real gap structure + player dropout (`src/external.py`, 2026-10-28)
Measured on the graded ATD frames: ball visible runs median 14 s (mean 25 s), ball gaps median 11 s (mean 18 s) and
**never 1-3 s** (sub-second or 3 s+); 20.5 of 22 players tracked per frame, 52% of frames complete, missing players come in
stretches (median 3 s, mean 10 s) of 1 (54%) to 8 players; 98% of lost player-frames are tracklet ends, not blinks.
`degrade_ball` now draws run lengths from the ATD percentile tables (old version: geometric 1-s gaps, the guess its own
comment admitted); `degrade_players` alternates complete stretches with stretches missing 1-8 players. Identity swaps are
not simulated: ATD slots never change team and the features are slot-agnostic, so only presence matters (PathCRF training
would need them; not done). Check on J03WMX after degradation: ball visible 0.597, 54% complete frames, 21.2 tracked.
All DFL CV numbers before this section used the old degradation; new ones below are not comparable to them.
HGB v3 under the realistic degradation: **0.438 / 0.828** (old yardstick 0.532 / 0.893); the grid still picks temp 1 /
switch 10 (temp 0.5 / switch 6 is 0.440, noise). Oracle dead balls would give 0.474 / 0.869, stoppage onsets are now late
by up to 2.3 s (p90) and 36 of 711 stoppages are missed -- the yardstick now shows the dead-ball lateness we see on
BAR-ATH, so a dedicated alive/dead model can be judged here (TODO item 2). Public-window numbers are optimistic relative
to the whole match: the ball is visible on 94% of its alive frames vs 59% over the graded match.
BiGRU under the realistic degradation: **0.464 / 0.839** (switch 3; old yardstick 0.585 / 0.904), still ahead of the HGB
(0.438 / 0.828). Combinations on the same folds: BiGRU H/A + HGB D 0.453 / 0.834; mean of the two log-prob sets
0.470 / 0.844 (switch 6) -- +0.006 over the BiGRU alone, inside fold noise, not adopted. (An earlier run of this
comparison mixed old-yardstick HGB folds with new-yardstick BiGRU folds and printed 0.564; discard.)
Deployed models retrained on the realistic degradation (2026-10-28): HGB public window 0.440 / 0.809 (old-degradation
model: 0.537 / 0.834), PathCRF + HGB dead 0.582 / 0.827 (was 0.622 / 0.853). The window has an almost complete ball, so
a model trained for 1-s gaps suits it better; whether that holds over the full match is what `TRAIN_DEGRADE=old` in
`eda/dead_diag.py` tests (train on the old degradation, test on the real one): **0.416 / 0.781** with 178 false
stoppages, vs 0.438 / 0.828 trained on the real one. Settled: the deployed HGB and BiGRU are trained on the realistic
degradation; the public window was flattering the old model because its ball is almost complete.
BiGRU retrained on it: public window **0.625 / 0.879**, 15 spells, dead-frame F1 0.968 (HGB 0.946), and as PathCRF's
dead-ball source **0.605 / 0.875** vs 0.582 / 0.827 with the HGB. On DFL its own dead balls were already at least as good
(BiGRU 0.464 vs BiGRU-H/A + HGB-D 0.453). **Dead-ball source switched to the BiGRU** (`pathcrf_run.py --alive` default,
`pod/pathcrf_setup.sh`); candidate file `evaluation/results/pathcrf.csv` = authors' PathCRF + BiGRU dead balls.

### Goalkeeper for PathCRF: already handled (2026-10-28)
PathCRF finds the keeper itself per episode (`datatools/utils.detect_keepers`: the player whose mean x is farthest from
the centre), in training and in inference. On our stitched tracks it picks a track 44.5 m from the centre at the median
(DFL: 41.3 m), present for the whole episode in every case, 5% of episodes below 30 m (DFL 2%). Nothing to add.

## Label transfer: learned PFF labels (`src/label_transfer.py`, 2026-10-28) -- PFF still does not help the frame models
Diagnosis: the hand rule that turns on-ball events into H/A/D, applied to DFL's *own* event stream, reproduces the official
DFL labels at only frame acc 0.89-0.92, spellF1 0.73-0.76, macroF1 0.88-0.92 (3 matches). So the rule, not PFF, was the
weak link. Teacher = HGB on the v3 tracking features (DFL ball given the real gap structure + moving dead ball, players
clean like PFF's) plus 9 event features (seconds since/until the nearest on-ball instant and its team, since/until
dead-type events, since period start), Viterbi switch 6 + min_dead. DFL leave-one-out vs official labels:
**frame acc 0.96, spellF1 0.752, macroF1 0.949** (v1 with a clean frozen ball: 0.756 / 0.953, but it labelled 0% of PFF
frames dead). PFF relabelled (`data_raw/external/pff_labels2_<gid>.npy`): dead fraction 0.30-0.41 (rule 0.34-0.70; the
rule over-deads some games), 516-644 label changes per match (DFL 460-613).

| HGB v3 training set (realistic degradation, DFL LOMO) | spellF1 / macroF1 |
|---|---|
| 6 DFL | 0.438 / 0.828 |
| 6 DFL + 64 PFF, rule labels | 0.429 / 0.829 |
| 6 DFL + 64 PFF, teacher labels | 0.431 / 0.831 |

Neutral: ten times more matches with decent labels do not move the frame model, so data volume is not its bottleneck.
PFF stays out of HGB/BiGRU training; it stays in for PathCRF (GPU ablation fold, trial 300). The teacher is kept
(`src/teacher.pkl`) for anyone who wants PFF labels in DFL's definition, e.g. for PathCRF's `ball_owning_team_id`.

## The task in one line
One LaLiga match (Barcelona vs Athletic, 2024-25 MD2). For each of the 199,319 frames of the
tactical video (25 fps) output `H` (Barça has ball), `A` (Athletic has ball), `D` (dead ball),
or `X` (abstain). Scored on the private full-match truth; we only see the first 5 min of play.

## Scoring — this drives everything
- Ranked on **spellF1**: each possession (maximal run of H or A) is one object. Predicted
  possessions are matched 1:1 to true ones by temporal IoU at thresholds 0.3..0.7, F1 averaged.
  Constant guess = 0. A possession 1 s late scores 0.65 on its own. Noise band ~0.04.
- **macroF1** (per-frame, mean of H/A/D F1) is the tie-break only.
- `X` is forward-filled with our previous label before spellF1 runs. So X inside a possession is
  free; X across a hand-over is as bad as being wrong.
- Baseline (nearest player to ball + 2 s majority vote) = spellF1 0.2433 full match. Beat that.
- Top 20 advances. Deadline Nov 17. Deliverables: CSV + public repo that regenerates it + short
  technical doc.

## Rules that bite
- No hand labels on this game, for anything (train, validate, fix). Pseudo-labels from an
  algorithm are fine. Hand-labelled OTHER games are fine.
- No event data / corrected tracking about this game from anywhere.
- **Do not upload the video or data to hosted LLM APIs** (terms of use). Local models only
  (whisper via mlx, local YOLO, etc). This also rules out Polyglot's Claude translate backend.
- Public submission repo must not contain data/ or derived frame-level exports.

## The data (all in `data/`, numbers verified locally)
| asset | what | notes |
|---|---|---|
| tactical ATD (`_atd.txt` + `.xml`) | 199,319 lines, `frame:p1;p2;..;p46:ball`, each `x,y,speed`, 0-1 normalised | row i = video frame i-1. 46 track *slots*, not players; slots 1-23 Barça, 24-46 Athletic (teamId in xml). Ball visible 56.9% all / 59.3% graded. ~20.5 players tracked per graded frame |
| tactical homography | 3x3 pitch(0-1)->pixel per frame, 100% valid | lets us project tracking onto the video |
| smart tagging xml | 638 phases (`BUILD UP`, `SET PIECES`...), seconds from video start, with `Team` tag | `Team` = N/A on all throw-ins. Phases, not possessions |
| broadcast ATD + homography | 164,488 frames, own timeline, 65.8% valid homography | not synced to tactical; has the commentary audio |
| videos | tactical 199,319 f; broadcast 164,488 f; ~13 GB | downloading into `data_raw/` |

Graded frames: 11185–90651 (1st half) and 110808–190197 (2nd half). 2nd half kicks off at
113742 — the 2,934 frames before that are graded and all `D`.

## What the public 5 minutes shows (`eda/out/timeline_public.png`)
- 21 segments / 17 possessions, mean 14 s, but possessions as short as **0.76 s and 1.3 s** exist
  (GT frames 15346-15364, 15766-15798). A 2 s majority vote erases those by construction.
- Nearest-player team agrees with GT on only 66% of alive frames with a ball: every pass flips it.
- Dead ball ≠ missing ball: in this window the ball is *visible* on 66% of D frames, and missing
  on 6.5% of alive frames. The baseline's "no ball → D" is the wrong proxy.
- Dead ball looks like: ball far from every player (nearest-player distance 10-15 m) and/or ball
  outside the pitch rectangle, with players walking. Smart tagging `SET PIECES` marks the restart.
- Smart tagging `Team` is roughly right at phase level but lags transitions by seconds and is
  blank during set-piece build-ups.

## Plan (proposal, individual week then converge)
1. **Feature table per frame** (numpy, cached in `eda/atd_tactical.npz` already): ball xy/speed,
   nearest player per team + distance, ball in/out of pitch, ball visible, smart-tag phase/team,
   n tracked players, team centroids / spread, ball velocity direction vs team.
2. **Dead-ball detector first** (D vs alive). It's the biggest easy win: ball off pitch, players
   static, SET PIECES tag, 2nd-half pre-kickoff. Whistle detection from audio if tactical video
   has audio (check `ffprobe` when the mp4 lands).
3. **Possession as a segmentation problem, not a per-frame classifier.** Detect *touch events*
   (ball speed/direction change near a player), assign each touch a team, then a possession
   changes when the opposing team gets N consecutive touches / holds for T seconds. Tune N,T on
   the whole-match *structure* (possession count, duration distribution look like football), not
   on the 5 minutes. HMM / Viterbi over {H,A,D} with touch-based emissions is the clean version.
4. **Ball gap filling**: interpolate short gaps with constant velocity; for long gaps, the player
   cluster motion tells you where play is.
5. Only then: broadcast video + commentary (Polyglot's mlx-whisper + Demucs path, local), needs
   a broadcast→tactical time alignment (match on player-cluster shape through both homographies).

## Commentary (Polyglot, local only)
Broadcast mp4 audio -> ffmpeg 16 kHz wav -> Demucs two-stems (vocals = commentators, no_vocals =
crowd + whistle) -> mlx-whisper large-v3-turbo with word timestamps -> `eda/audio/commentary_*.jsonl`.
Times are broadcast seconds; mapping to tactical frames needs the broadcast->tactical alignment.
Only Polyglot's mlx-whisper + Demucs path is allowed; its Gemini-audio and Claude backends upload
the audio and are off limits for this data.

**Done (2026-10-25):** `~/Polyglot/.venv/bin/python eda/commentary.py --source broadcast` ran on
the full match in ~40 min (Demucs ~30 min, Whisper ~8 min). English commentary, 5,020 segments /
18,010 words with word timestamps -> `eda/audio/commentary_broadcast.jsonl`; stems in
`eda/audio/stems/htdemucs/broadcast/{vocals,no_vocals}.wav` (1.2 GB each, gitignored; `no_vocals`
is the crowd + whistle track for a whistle detector). Mostly colour commentary: play-by-play words
are sparse (corner 21, offside 13, foul 19, goal 88, throw 4, free kick 1 mentions) so it will be a
restart/stoppage cue, not a frame-level possession signal. Timestamps map to tactical frames with
`eda.align.b2t` (see alignment section).

## databallpy
`eda/dbp.py` builds a real databallpy `Game` from the numpy cache (its own loaders reject this
export). Everything on https://databallpy.readthedocs.io/en/latest/getting_started/visualizations_page.html
then works: `plot_tracking_data`, `save_tracking_video`, `add_velocity`, `get_pitch_control`.
Placeholders to replace with our model's output: `ball_status` (= ball tracked) and
`player_possession` (= nearest player). Team colours default to green=home, red=away; pass
`team_colors=["#d62728", "#1f77b4"]` to match `eda/render.py`.
```python
from eda.dbp import game
g = game(); td = g.tracking_data
td.add_velocity(g.get_column_ids() + ["ball"], filter_type="savitzky_golay", max_velocity=50)
from databallpy.visualize import plot_tracking_data, save_tracking_video
plot_tracking_data(g, 11260, add_velocities=True, add_player_possession=True)
save_tracking_video(g, 11185, 11685, "eda/out", title="clip", add_velocities=True, add_player_possession=True)
```

## Videos
`data/tactical/BAR-ATH_tactical.mp4` (5.2 GB, from the team bucket `s3://more-than-a-hack-2027`,
profile `mtah`). 199,319 frames at 25 fps, **has a stereo AAC audio track** -> whistle / crowd
detection on the submission timeline with no alignment problem. **Checked: the track is near-silent**
(mean -58 dB, peak -44 dB; whisper hallucinates "Thank you" on it) -> no commentary, probably no
usable whistle either. Commentary comes from `data/broadcast/BAR-ATH_broadcast.mp4` (13 GB, same
bucket, `broadcast/` prefix).

## Running things
```
.venv/bin/python eda/atd.py                 # builds/validates the numpy cache (5 s)
.venv/bin/python eda/plot.py 11185 15889    # pitch PNGs -> eda/out/
.venv/bin/python eda/plot.py --clip 11185 11600   # GIF
.venv/bin/python eda/timeline.py            # the 5-min comparison figure
.venv/bin/python eda/render.py --source tactical  --start 11185 --end 11685          # pitch-view mp4
.venv/bin/python eda/render.py --source tactical --start 11185 --end 11685 --video --possession --trail   # camera + overlay | pitch
.venv/bin/python eda/render.py --source tactical --start 11185 --end 18685 --velocities --possession --trail --heatmap   # all overlays
.venv/bin/python eda/dbp.py                 # databallpy smoke test -> eda/out/dbp_*
~/Polyglot/.venv/bin/python eda/commentary.py --start 600 --end 900                  # local whisper on broadcast audio
.venv/bin/python starter/baseline.py        # regenerate the 4 baselines
.venv/bin/python evaluation/run_evaluation.py   # score everything in evaluation/results/
```

## External data (allowed: RULES.md "External data about other games", FAQ "must be about other games")
Hard line: **nothing human-made about BAR-ATH itself** — so ESPN/FotMob/Opta play-by-play *of this
match* is out. Anything from other matches is fine as long as it is public and documented in the
tech doc (source, who labelled, how much). Candidates, best first:

| source | what | why we care |
|---|---|---|
| DFL / IDSSE open dataset (CC-BY 4.0, 7 Bundesliga matches 2022/23, 25 Hz, ~2.4 GB) | tracking **+ per-frame `possession` (team) and `ballstatus` (alive/dead) codes** + events; `pip install floodlight`, `IDSSEDataset().get(match_id)` | exactly our label space on real tracking -> train/validate the tracking->H/A/D model |
| Metrica sample-data (3 games, tracking+events synced, 105x68, same provider as ours) | events incl. passes/recoveries/ball lost per team | derive team possession per frame; same coordinate convention as our ATD |
| PFF FC 2022 World Cup (64 matches, broadcast tracking + events, free with signup) | possession events + broadcast-derived tracking | closest analogue to our *broadcast* ATD |
| SkillCorner open data (10 A-League matches, broadcast tracking, "Dynamic Events", phases of play) | no video, no events | broadcast-tracking gap behaviour |
| SoccerNet-Echoes (github SoccerNet/sn-echoes) | Whisper ASR transcripts of broadcast commentary per game, JSON {start,end,text}; SoccerNet video/labels need the SoccerNet NDA | commentary -> event/possession language model; SoccerNet Ball Action Spotting (12 classes incl. pass/drive/throw-in/out/free kick) gives event timestamps to align text to |
| GOAL (arXiv 2211.04534) | ~20 games, audio + human transcripts of live commentary | small, human-quality commentary |

Note: the tactical ATD smart-tagging + our own algorithmic pseudo-labels on BAR-ATH are also fine
(no human decided them).

## Broadcast -> tactical alignment (done, `eda/align.py`)
The LaLiga scoreboard clock is OCR'd (tesseract, 1 crop / 2 s, 80% of samples carry a clock;
the rest are replays/cutaways/pre-match). Clock runs in real time, so each half is one offset:

| half | clock = broadcast_sec + | tactical frame = | broadcast range with clock |
|---|---|---|---|
| 1 | -286.86 s | 11185 + 25·clock | 308–2978 s |
| 2 | -772.90 s | 113742 + 25·(clock-2700) | 3492–6544 s |

Refined to the frame (2026-10-26) by minimising the broadcast-ATD vs tactical-ATD ball
disagreement: clock-only offsets (-286.5 / -773.5) gave 2.4 / 4.0 m median disagreement, the
refined ones 0.82 / 0.86 m. Both values are kept in `eda/align.json` (`offset_clock`, `offset`).

Broadcast ball fusion was tried and dropped: the broadcast ATD carries a pitch-coordinate ball on
only 26% of graded frames and adds just 2.6% of frames the tactical feed does not already see.

91–94% of reads agree with the offset to within 1 s; the rest are OCR misreads. Validated
independently: broadcast-ATD vs tactical-ATD player-centroid distance is minimised within ±1–2 s of
the mapped frame. Uncertainty ≈ ±0.5 s (12 frames) from clock granularity. The clock display changes
format in stoppage time (reads stop at 44:50 / ~96:00) but the offset still applies there.
Use: `from eda.align import b2t; b2t(broadcast_seconds) -> tactical frame or None`. Constants in
`eda/align.json`; reads in `eda/out/ocr/clock_ocr.csv`. Tactical kickoffs confirmed from the ball
leaving the centre spot at frames 11185 and 113742.

## Possession v1 (`src/possession.py`, 2026-10-26)
Ball gap fill (≤2 s linear) -> per-frame evidence (H/A: `(8 m - dist to nearest player of that
team)`, D: off pitch / outside halves / ball missing / ball lying still away from everyone) ->
Viterbi over {H,A,D} with a switch penalty. numpy only, ~1 s for the whole match.
Tuned on the **7 DFL open matches** (real per-frame possession + ball-status labels) and **Metrica
sample games 1+2** (labels derived from their event stream), ball hidden to BAR-ATH-like 35%
visibility (`src/external.py`), never on the public window. DFL conversions cached in
`data_raw/external/dfl_*.npz` (12 s each to build with floodlight).

| | spellF1 | macroF1 | spells (gt) |
|---|---|---|---|
| 7 DFL matches degraded, organisers' baseline_smooth50 | 0.12 | 0.44 | 725 (437) |
| 7 DFL matches degraded, **v1** | **0.32** | 0.69 | 322 (437) |
| Metrica 1+2 degraded, baseline_smooth50 | 0.13 | 0.48 | 470 (346) |
| Metrica 1+2 degraded, **v1** | **0.44** | 0.67 | 309 (346) |
| BAR-ATH public 5 min, baseline_smooth50 (official scorer) | 0.33 | 0.62 | 26 (17) |
| BAR-ATH public 5 min, **v1** (sanity only) | **0.35** | 0.67 | 24 (17) |

Smart-tagging SET PIECES windows are weak D evidence only: 56% of their frames are dead in the
public GT, and the tracked ball is never "still" inside them (median 5 m/s, same as open play), so
the restart moment cannot be read off ball speed.

Still under-predicting spell count (misses short possessions) and D is rule-only. Next (v2):
train a per-frame classifier on DFL features (distances, ball speed, player spread, visibility) ->
P(H/A/D) as Viterbi emissions; touch-based switches; whistle (crowd stem) + commentary cues via
`eda.align.b2t` into the D evidence.

**S3:** my IAM user (`Eli-more-than-a-hack`) has no `s3:PutObject` on the bucket, so external data
is local only (`data_raw/external/`, gitignored) until Brandon grants write. Planned layout:
`s3://more-than-a-hack-2027/external/{metrica-sample,dfl}/`.

## Possession v2 (`src/v2.py`, `src/features.py`, 2026-10-26)
Per-frame HistGradientBoosting classifier (38 slot-agnostic features: per-team nearest distances
and counts, ball visibility/gap length, ball speed, pitch position, player spread/speed, ±0.5 s and
±2 s rolling context) trained on the **7 DFL matches** with the ball degraded, home/away mirrored
(2.0M rows). log P(H/A/D) are the Viterbi emissions, switch penalty 10. Model: `src/v2_model.pkl`
(3 MB, regenerate with `--train`, ~2 min). BAR-ATH inference: `python src/v2.py` (~30 s).

| | spellF1 | macroF1 | spells (gt) |
|---|---|---|---|
| DFL leave-one-match-out, v1 | 0.32 | 0.69 | 322 (437) |
| DFL leave-one-match-out, **v2** | **0.50** | **0.89** | 317 (437) |
| Metrica 1+2 degraded (never trained on), v1 | 0.44 | 0.67 | 309 (346) |
| Metrica 1+2 degraded (never trained on), **v2** | **0.48** | 0.70 | 342 (346) |
| BAR-ATH public 5 min, baseline_smooth50 | 0.33 | 0.62 | 26 (17) |
| BAR-ATH public 5 min, v1 | 0.35 | 0.67 | 24 (17) |
| BAR-ATH public 5 min, **v2** (sanity only) | **0.52** | **0.70** | 16 (17) |

Frame accuracy on held-out DFL is 0.90; the DFL->Metrica gap (0.89 -> 0.70 macroF1) is partly the
derived Metrica labels, partly domain shift. Next: whistle + commentary cues as extra features
(via `eda.align.b2t`), smart-tagging windows as a feature, bigger context windows, and a
transition model that knows a possession starts with a controlled touch.

## Audio cues: negative result (`src/audio_cues.py`, 2026-10-26)
Whistle detector on the Demucs crowd stem (3-4.5 kHz tonal peak, robust z > 6, 227 onsets) and
commentary restart words (throw-in / corner / free kick / foul / offside / penalty / goal, word
timestamps mapped with `eda.align.b2t`). Checked against the smart-tagging SET PIECES windows
(no labels involved): P(set-piece window | whistle) 0.26 vs base 0.22, restart words 0.27. In the
public window P(D | whistle) is 0.14 vs base 0.20 -- the stadium crowd whistles constantly, so the
detector finds fans, not the referee. Not used in v3. Would need a referee-whistle classifier
trained on other games' audio (e.g. SoccerNet-Echoes games) to be worth revisiting.

## Possession v3 (`src/v2.py` + 8 "since/until" features, 2026-10-26)
Adds seconds since/until the last home touch, away touch, ball off pitch and ball speed jump
(46 features). ±5 s rolling context was also tried: no gain, dropped.

| | spellF1 | macroF1 | spells (gt) |
|---|---|---|---|
| DFL leave-one-match-out, v2 -> **v3** | 0.50 -> **0.56** | 0.89 -> **0.91** | 317 -> 316 (437) |
| Metrica 1+2 degraded (held out), v2 -> **v3** | 0.48 -> **0.52** | 0.70 -> **0.71** | 342 -> 320 (346) |
| BAR-ATH public 5 min, v2 -> v3 (sanity only) | 0.52 -> 0.39 | 0.70 -> 0.69 | 16 -> 27 (17) |

Split result: every external check improves, the public window gets worse and v3 fragments
BAR-ATH into 27 spells. 17 possessions is far too few to decide, but the fragmentation is a real
hint that the touch features behave differently on our tracklets (46 slots, ~20.5 tracked, ID
switches) than on DFL's clean 22 tracks. **v2 stays the submission candidate** until the external
validation also degrades the *player* tracking to BAR-ATH-like quality (drop ~7% of player
observations, add ID churn) -- that is the honest way to pick between them.

## PathCRF (KDD 2026, github.com/hyunsungkim-ds/pathcrf) — first honest number (2026-10-26)
Cloned to `~/pathcrf` (own venv: torch, torch_geometric, rdp). Its training data *is* the DFL open
dataset (raw XML linked in from floodlight's cache, `datatools/preprocess.py` run). Authors' pretrained
checkpoint `saved/140` (leave-one-match-out, held-out = J03WR9) run on that match, 70 s on CPU.
It predicts the possession edge (carrier -> receiver) from **player tracking only, no ball**.

| J03WR9 held out, GT dead frames substituted for both | spellF1 | macroF1 | spells (gt 370) | team acc alive |
|---|---|---|---|---|
| GBDT v3 trained on the other 6 | 0.57 | 0.93 | 278 | 0.90 |
| **PathCRF** pretrained | **0.69** | **0.95** | **366** | **0.93** |

PathCRF gets the possession *structure* right (366 vs 370 spells) where our model merges short
possessions. It does not model dead balls -> pair it with our D detector. Open problem: it wants
11 identified players per team; BAR-ATH has 46 tracklet slots with ID churn (see src/pathcrf_io.py
once written). One-line patch applied in `~/pathcrf/inference.py` (pandas read-only arrays).

### PathCRF on BAR-ATH (2026-10-26)
`src/pathcrf_io.py` stitches the 46 tracklet slots into 11 persistent tracks per team (segments
linked in start order within 5 s / 8 m/s, gaps <= 2 s interpolated, position held inside an
episode), alive/dead from v2's CSV, episodes >= 10 s cover 100% of alive time, 10.8 players per
team per frame. `src/pathcrf_run.py` runs the authors' checkpoint (~50 s CPU) and merges: PathCRF
team on alive frames it covers, v2 elsewhere. Public window (sanity): **spellF1 0.615, macroF1
0.716, 24 spells** vs v2 0.521 / 0.698. No training on our side yet; goalkeeper identity not
provided (model expects it). `src/commentary_model.py`: stoppage classifier trained on
SoccerNet-Echoes transcripts + SoccerNet labels (other games), applied through the alignment.

## Commentary stoppage model (`src/commentary_model.py`, 2026-10-27)
TF-IDF + logistic regression on SoccerNet-Echoes English transcripts (330 matches with labels,
352k 10-s windows) vs SoccerNet stoppage labels within a lag window. Held-out games: AUC 0.565,
AP 0.51 (base 0.44); top terms are the right ones (throw in, corner, offside, foul, into touch).
On BAR-ATH through the clock alignment it covers 72% of frames; public window P(D | p >= 0.5) =
0.29 vs base 0.20 (keyword rule was 0.23). Weak but real. Not in v2/v3 (no such feature on DFL);
usable only as extra D evidence at decode time with a hand-set weight -> parked, documented.
Output: `eda/out/commentary_stoppage.npz` (per-frame probability).

## Ball detector (`src/ball_detect.py`, 2026-10-27)
YOLOv8n trained on 2,269 ball tiles + 3,880 ball-free tiles cut from every 50th tactical frame,
labels = ATD ball projected through the homography and snapped to the brightest small blob
(median snap 14 px). No human label. Val tiles: P 0.76, R 0.52, mAP50 0.58. Every-50th-frame
check against the ATD (3,119 alive frames):

| conf | detects on ATD-ball frames | median error vs ATD | detects on ATD-missing frames |
|---|---|---|---|
| 0.10 | 97% | 1.5 m | 75% |
| **0.25** | 87% | 1.3 m | **46%** |
| 0.40 | 76% | 1.2 m | 30% |

At 0.25 the fused ball (`src/ball_fuse.py`: ATD first, detector otherwise) should take graded-frame
visibility from 59% to ~78%. Inference over the whole match ≈ 2.5 h on the laptop (MPS), 4x2 tiles
of 640 px per frame. Weights `src/ball_yolov8n.pt` (not in git since 2026-11-01: bucket `derived/ball_yolov8n.pt`); detections `eda/out/ball/detections.npz` and
bucket `derived/ball_detections.npz`. Next: re-run v2 / PathCRF stitching on the fused ball and
retrain v2 with DFL ball degraded to ~78% instead of 57%.

## Ball fusion result (2026-10-27): negative, parked
Full-match inference: 167,896 detections (2 h 23 min, MPS). Against the ATD ball on frames that
have one, gross errors are common: >5 m off for 42% of detections at conf 0.25-0.4, 24% at
0.6-0.8, 7% above 0.8. A temporal-support filter (`ball_fuse.support`: neighbours within ±5
frames that lie within 1.5 m/frame) separates real tracks from jumps, but even 10/10-consistent
detections are >5 m off 12% of the time, and 80% of fills on ATD-missing frames sit in long
gaps (dead-ball stretches) with no ATD ball within 2 s to check against.

Public window (sanity only; it already has an ATD ball on 88% of frames, fills touch 7%):

| ball | DFL train visibility | spellF1 | macroF1 | D frames predicted (GT 1469) |
|---|---|---|---|---|
| ATD only (v3) | 0.57 | 0.391 | 0.686 | 466 |
| ATD only | 0.78 | 0.400 | 0.676 | – |
| fused, conf 0.25 | 0.57 | 0.400 | 0.602 | 76 |
| fused, conf 0.25 | 0.78 | 0.372 | 0.561 | – |
| fused, conf 0.5 + support 10/10 | 0.57 | 0.373 | 0.656 | – |

Fills land on dead-ball frames (17% of GT-dead frames get one) and turn D into H/A; the dead
class collapses. Training visibility is irrelevant (0.57 vs 0.78 within noise). Conclusion: the
detector's precision is not good enough to feed the ball features; same verdict as the
broadcast-ball fusion. Kept: `src/ball_fuse.py` (support filter, `--source tactical_fused` on
`src/v2.py` and `src/pathcrf_io.py`), weights and detections in the bucket, for anyone who wants
to retrain the detector with more data (SoccerNet Tracking) and try again.

## BiGRU sequence model (`src/seq.py`, 2026-10-27)
BiGRU (hidden 64, 30-s windows) over the 46 v3 per-frame features, trained on the 7 DFL matches
(ball degraded to 0.57, H/A mirrored), 8 epochs, log-softmax as Viterbi emissions. Leave-one-
match-out on DFL (pod, CPU, ~3 h): **spellF1 0.603 / macroF1 0.914** at switch 3 (0.598 at 6,
0.590 at 10), vs HGB v3 0.56 / 0.91 and v2 0.50 / 0.89. Frame accuracy 0.92-0.94 per match.
BAR-ATH public window (sanity only): switch 3 -> 0.474 / 0.674, 21 spells; switch 6 or 10 ->
0.500 / 0.669, 19 spells (GT 17). Unlike v3 it does not fragment our match. Model committed as
`src/seq_model.pt`; `--train` regenerates it in ~25 min on the laptop. Result files
`evaluation/results/seq.csv` (switch 3, the DFL-chosen value) and `seq_sw{6,10}.csv`.
Also: DFL CV with ball visibility 0.78 instead of 0.57 for the HGB: 0.558 / 0.910, i.e. unchanged.
PathCRF with the BiGRU's dead labels instead of v2's (`pathcrf_run.py --alive seq.csv`, absolute
paths, pathcrf venv): public window 0.587 / 0.701, 28 spells vs 0.615 / 0.716, 24 spells with v2.
Within noise; v2 stays the dead-ball source for PathCRF.

## PFF FC World Cup 2022 (`src/pff.py`, 2026-10-27)
64 matches, tracking 29.97 fps (11 players per team on every frame, ball missing on 26% of frames,
mostly dead-ball) + event data, public, downloaded from PFF's Drive. Converter: 15 s per match ->
`data_raw/external/pff_<gid>.npz` at 25 fps (same dict as the DFL loader) and
`~/pathcrf/data/pff/tracking_processed/<gid>.parquet`. Labels are algorithmic: D from the period
start to the first on-ball event, from an OUT / foul / goal event to the next on-ball event, after
END; else the team of the last on-ball (OTB) event. 35% dead in 93-125 spells of 16-23 s (DFL: 42%,
92-103 spells of 23-27 s). Label quality vs DFL: on clear-possession frames (ball within 1.5 m of a
player) the label matches the nearest player's team on 89% (DFL 92%); half of the frames the rule
calls dead look like live play (ball visible, players at 2.3 m/s), so the dead rule over-reaches.

DFL-trained models on PFF 10502 (its labels as truth): HGB v3 0.27 / 0.67 (frame acc 0.66), BiGRU
0.36-0.38 / 0.73 (0.74), PathCRF checkpoint 0.25 / 0.80 with 1019 spells (gt 394).

**Training the HGB on PFF is a negative result.** Held-out DFL match, HGB v3 features, PFF every 5th
frame + mirror:

| train | J03WR9 spellF1 / macroF1 | J03WMX |
|---|---|---|
| 6 DFL | 0.562 / 0.925 | 0.547 / 0.917 |
| 6 DFL + 64 PFF | 0.545 / 0.917 | 0.504 / 0.907 |
| 64 PFF only | 0.368 / 0.766 (frame acc 0.78) | 0.355 / 0.794 (0.80) |

The possession definition behind my PFF labels is too far from DFL's (and so from the organisers')
to help; a better rule would need to be tuned against DFL. Kept for PathCRF, which learns from the
on-ball player ids (PFF's reliable part) inside alive episodes: `pod/vast.sh` trains DFL-only and
DFL+PFF leave-one-out folds on a rented GPU and the two get compared on DFL and on our match.
