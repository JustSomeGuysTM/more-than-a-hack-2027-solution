# Team meeting 2026-11-01: World Cup data, SoccerNet, task split

## Why the World Cup data is not in the submission

**The World Cup games have no possession labels.** PFF ships tracking and event data, not the per-frame
"who has the ball" label we are scored on. We made the labels ourselves with a rule: dead from a stoppage
event until the next touch, otherwise the team of the last touch. That rule is roughly right, but about half
the frames it calls dead look like live play. So those 64 games teach the model a slightly different
definition of possession than the organisers use, and the score (spellF1) punishes exactly that kind of
boundary disagreement.

**It did help in the fair test.** On one held-out German match, training with the World Cup games beat
German-only at the same training length (trial 300: 0.639 / 0.940 vs trial 200: 0.608 / 0.933). That was
the ablation that made it look promising.

**The final run was not a fair fight.** With the World Cup games each training pass took almost two hours,
so the box only had time for 15 passes before the credit ran out. The German-only run got 50 passes. The
longer run won on both checks, so we shipped it.

| model | German validation match (J03WR9) | our match, public window (sanity only) |
|---|---|---|
| trial 401: German only, 50 passes (submitted) | 0.707 / 0.954 | 0.660 / 0.884 |
| trial 400: German + World Cup, 15 passes | 0.683 / 0.947 | 0.556 / 0.855 |

Honest summary: "more data at short training lost to less data at long training." We never ran more data
at long training. That is trial 402, the one experiment that could still move the Phase 1 score. The other
lever is label quality: `src/label_transfer.py` (the "teacher") matches the German labels much better than
the rule (spellF1 0.75 vs ~0.73, and a sane dead fraction), and its labels exist for every World Cup game
(`data_raw/external/pff_labels2_<gid>.npy`), but they were never wired into the parquets PathCRF reads.

## What SoccerNet is

SoccerNet Tracking 2023 is a public set of short broadcast clips from the Swiss league, each with
human-drawn boxes around every player, referee and the ball in the image. No pitch coordinates, no
possession labels. Its only use for us is training a ball detector. Our current detector
(`src/ball_detect.py`) was trained on boxes auto-generated from our own match, and it was too imprecise to
help: 42% of detections were more than 5 m off, and the fills landed on dead-ball frames and wiped out the
dead class. Real ball boxes could fix that. It is allowed (public, other games, documented in the tech doc).
It is worth nothing for the Phase 1 score unless cross-validation says otherwise, but it is a good
innovation-round asset and a self-contained project. The train set is in the bucket at
`s3://more-than-a-hack-2027/external/soccernet/SN-Tracking-2023/`.

## Task split

1. **Retrain (start this week).** Re-rent the box, run `pod/vast.sh pack` / `setup`, launch trial 402 with
   `SOURCES="sportec pff"`. Before launching, swap the rule labels for the teacher labels in the PFF parquets
   (`ball_owning_team_id` in `src/pff.py`): biggest quality lever, zero GPU cost. Use `N_EPOCHS=30`, batch 64
   first (~53 h, ~$25). Adopt only if it beats trial 401 on **both** the German validation match and the
   public window. Exact commands: `eda/NOTES.md` item 0 and `pod/HANDOVER.md`.
2. **SoccerNet ball detector (no deadline pressure).** Unzip the train set, pull the ball boxes from the
   per-clip `gt.txt`, fine-tune `src/ball_yolov8n.pt`, rerun the every-50th-frame check against the tactical
   ball (table in `eda/NOTES.md`, "Ball detector"). Judge any pipeline change with `simulate_detector` on
   DFL cross-validation, never on the public window. Pitch this as the innovation-round piece.
3. **QA video and submission.** Side-by-side video: tactical feed with the predicted label overlaid,
   tracking plot beside it, spell changes marked. Then the clean-clone reproduction with
   `pod/pathcrf_setup.sh`, `python evaluation/validate_submission.py`, rename to `<team_name>.csv/.md`,
   make the repo public without `data/`, submit the form. Rule: watching the video to sanity-check is fine;
   nobody may hand-correct labels on this game from it.

## Two cautions

- The public five-minute window is a sanity check only. Tuning on it overfits; the yardstick is the
  held-out German match.
- Only task 1 can change the score. If people are short on time, it gets priority over the other two.
