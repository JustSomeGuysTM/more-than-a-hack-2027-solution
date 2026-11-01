# Phase 1 task split (2026-11-01)

From the 2026-11-01 meeting (`meeting.md`). Deadline **Nov 17**; the file should be locked around Nov 10-12 so the
clean-clone reproduction can be checked before the form.

| task | owner | plan | can it move the Phase 1 score? |
|---|---|---|---|
| Trial 402: PathCRF on German (DFL) + World Cup (PFF) data | Will | [402_retrain.md](402_retrain.md) | yes, the only one that can |
| SoccerNet ball detector | Gurashish, with Eli | [soccernet_ball_detector.md](soccernet_ball_detector.md) | only if its ceiling check says so; otherwise the innovation-round piece |
| QA plots and videos | Eli | [qa_plots.md](qa_plots.md) | no, but it shows where the score is lost |
| Submission and public repo | Brandon | [submission_checklist.md](submission_checklist.md) | protects it: a file the repo cannot regenerate is not a valid submission |

Two rules apply to everyone:
- No hand labels or hand corrections on BAR-ATH, ever. Watching the video to debug is fine.
- The public five minutes are a sanity check. Model choices are made on held-out DFL matches.
