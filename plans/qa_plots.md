# QA plots and videos — Eli

Goal: see where the score is lost, and check the submission is not doing anything silly. Watching is allowed; turning
what we see on BAR-ATH into labels, corrections or frame-specific rules is not (RULES.md, rule 1). Everything rendered
from BAR-ATH stays in `eda/out/` (gitignored) and is never uploaded anywhere.

## What exists

- `eda/render.py`: pitch view as video; `--video` puts the real camera frame beside it with the tracking projected
  through the homography; overlays for velocities, a nearest-player "possession" placeholder, trails, pitch control.
- `eda/plot.py`: single frames or short GIFs with the public GT label.
- `eda/timeline.py`: the public 5 minutes as strips (GT, ball visibility, nearest-player team, smart tagging).
- `eda/dead_diag.py`: dead-ball error numbers on DFL folds (onset/offset lag, missed stoppages).

None of them show our predictions, and none work on DFL or PFF.

## Where the truth is

| data | ground truth | our predictions without leakage | video |
|---|---|---|---|
| DFL, 7 matches | official per-frame possession and ball status | yes: PathCRF folds 200-206, each on its held-out match; BiGRU out-of-fold dead balls from `src/seq.py --cv` (`eda/out/seq_folds.npz`) | none as far as we know (open data is tracking + events) |
| PFF WC22, 64 matches | none official: our rule labels and the teacher's | only the one PFF validation match of 400 | check the PFF drive; probably none |
| BAR-ATH | the public 5 minutes | yes, it is the test set | tactical and broadcast mp4 |

DFL is where QA can be quantitative. BAR-ATH is a sanity check.

## Status (2026-11-02)

Steps 1-6 done; findings in `eda/NOTES.md`, section "QA (DFL out of fold, PFF labels, BAR-ATH checks)". Headline: the
team call costs far more than dead balls (team oracle 0.589 -> 0.904, dead-ball oracle -> 0.615); teacher PFF labels
look right in the clips; BAR-ATH checks pass. Clips for the team: `aws s3 sync s3://more-than-a-hack-2027/qa/ eda/out/qa/
--profile mtah` (DFL and PFF only; regenerate with `bash eda/qa_batch.sh`). BAR-ATH videos stay local.

## Status (2026-11-01)

`eda/qa.py` implements steps 1-5; `src/pathcrf_run.py --dfl <match> --trial <t> --save` writes the PathCRF fold predictions
it reads. Commands are in the `eda/qa.py` docstring. `python3 eda/qa.py selftest --render` checks the error typing and
the oracles on synthetic labels and renders a synthetic clip; it passes. Nothing has been run on real data yet.

What each command gives:
- `dfl`: per-match spellF1/macroF1 for the full pipeline, BiGRU alone and PathCRF with GT dead balls, plus the three
  oracle fixes (stoppage onsets on time, dead balls exact, team exact); error counts by type; `errors.csv`;
  `timeline_<match>.png`.
- `clips --type <type>`: the worst N events of a type as 2D clips (degraded ball as the BiGRU saw it, true ball as a
  dotted ring where hidden), label strips for GT / ours / BiGRU / PathCRF.
- `pff <gid>`: rule vs teacher dead fraction, possession counts, agreement, the dead/alive disagreements >= 2 s, a
  timeline, optional clips.
- `barath`: pipeline-bug checks on the submission CSV (label shares per half, X on graded frames, kick-off and
  half-time), then a video of every label change (2 s before, 3 s after), optionally beside the tactical camera.

## Steps

1. **Out-of-fold predictions on DFL** (small code). Add `--save` to `src/pathcrf_run.py --dfl` to write per-frame labels
   for each held-out match (`eda/out/oof/`). Build two variants: the full pipeline (BiGRU out-of-fold dead balls,
   decoded from `seq_folds.npz` as `src/seq.py` does, plus PathCRF's team on alive frames) and the current `--dfl`
   behaviour (GT dead balls). The gap between them is the dead-ball model's cost.
2. **Error timeline per match.** One PNG per DFL match: strips for GT, prediction, ball visibility after degradation and
   PathCRF coverage, then the disagreements typed:
   - dead-ball onset late/early, restart late/early;
   - missed stoppage, spurious stoppage;
   - missed possession change, spurious change (a spell split in two);
   - wrong team for a whole spell.
   For each type, count it across the 7 matches and rescore with that type fixed from GT (an oracle swap). That says
   where the missing spellF1 goes, which is the most useful single output of this task.
3. **Error browser.** The worst N segments of each type as 10-20 s pitch clips (±5 s context): generalise the
   `eda/render.py` pitch panel to the DFL/PFF caches in `data_raw/external/`, with GT and prediction as label strips
   under the pitch and possession changes marked.
4. **PFF label check (feeds Will's 402).** For a few PFF games: rule labels vs teacher labels vs PFF's events on one
   timeline, plus clips where they disagree on dead/alive. Shows whether the teacher really fixes the dead stretches
   before 402 trains on them.
5. **BAR-ATH QA video** (meeting task 3). `eda/render.py --video` with the tactical camera, the pitch panel, a label strip
   for the submission CSV (public GT under it for frames 11185-18684), spell changes marked, and the current label in
   large letters. Either the whole match at `--step 5`, or an index of every predicted spell change (2 s before, 3 s
   after), so someone can scan ~25 changes per half in minutes. It is for pipeline bugs (a half flipped, a frame offset,
   a block of X), not for fixing individual labels.
6. Write findings in `eda/NOTES.md` under a "QA" section. Anything that suggests a model change is tested on DFL CV,
   never applied straight to BAR-ATH.
