# More Than A Hack 2027 — handover (written 2026-10-28, keep this if the Claude session dies)

## The competition
- Metrica / FC Barcelona "More Than A Hack 2027", Phase 1. Team: Elijah Cavan, Gurashish Bagga, Brandon Hiles, William Ngana.
- **Deadline 2026-11-17.** Top 20 advance. Submit via the Google Form from the organisers' email: `<team_name>.csv`, public repo URL, `<team_name>.md` (or .pdf) tech doc. One response per team, last one counts.
- Task: one label per frame of the BAR-ATH tactical video (199,319 rows, `frame,label`, labels H/A/D/X). Ranked on **spellF1** (possessions as whole objects), macroF1 tie-break. Graded frames 11185-90651 and 110808-190197.
- Rules that bite: **no hand labels on this game for any purpose**; no human-made data about BAR-ATH from anywhere (ESPN/FotMob play-by-play etc. forbidden); external data must be about other games, public, and listed in the tech doc (source, who labelled, volume); the public 5-minute ground truth is for sanity only, never for tuning; **never upload the match video/data to a hosted LLM** (terms of use); the public repo must not contain `data/` or frame-level exports. Algorithmic pseudo-labels on BAR-ATH are fine.
- Full rules in the repo: `RULES.md`, `SUBMISSION.md`, `FAQ.md`, `EVALUATION.md`.

## Where everything lives
- Laptop: `~/mtah2027` (clone of the team repo; `.venv` python 3.11; `data/` unpacked, `data_raw/` downloads, both gitignored). PathCRF code at `~/pathcrf` (own `.venv`; our 3-file patch in `~/mtah2027/pod/pathcrf.patch`).
- Team notes (the real explainer, read this first): `~/mtah2027/eda/NOTES.md`. Tech doc draft: `~/mtah2027/submission/TECH_DOC.md`. Data sources: `eda/DATA_SOURCES.md`.
- Team repo (private): https://github.com/JustSomeGuysTM/more-than-a-hack-2027-solution — local git remote is called `team`.
- **PR #6** (branch `possession-v4` -> main) holds all work since Oct 27 and is **open, not merged**: https://github.com/JustSomeGuysTM/more-than-a-hack-2027-solution/pull/6 . PR #5 was merged Oct 27. PRs #2-#4 were closed as superseded.
- S3 bucket `s3://more-than-a-hack-2027` (AWS profile `mtah`): `external/dfl_*.npz`, `external/pff/` (raw World Cup drive), `external/pff_<gid>.npz`, `external/soccernet/` (Echoes, Labels, Tracking-2023: train.zip uploaded; test/challenge were still downloading), `derived/` (models, detections, results).
  - `aws s3 ls s3://more-than-a-hack-2027/ --profile mtah`

## GitHub: two accounts
- Work account `ecavan95` is the default. The team repo is under the **personal** account `ecavan`.
- `gh auth switch -u ecavan` before touching the team repo with `gh` (and `gh auth switch -u ecavan95` to go back). `git push team possession-v4` works once switched.
- The PathCRF weights will be a GitHub **release** named `pathcrf-weights` on the team repo (files `<trial>.tgz`). Download: `gh release download pathcrf-weights -R JustSomeGuysTM/more-than-a-hack-2027-solution -p 400.tgz`.

## The vast.ai GPU box (paid, ~$45 credit, ~$0.46/h)
- Instance: Texas, 1x RTX 4090, 51 GB disk. Account dashboard: https://cloud.vast.ai/instances/ (Instances page shows the SSH command; the key icon shows it too).
- **SSH:** `ssh -i ~/.ssh/vast -p 21901 root@ssh4.vast.ai` (throwaway key pair `~/.ssh/vast`, public key registered on vast.ai; if the port/host changed, read it off the Instances page).
- What runs there: `~/pathcrf/queue.sh` (detached with setsid; survives disconnects). Order: trial 200 (DFL fold 0, done) -> 201-206 (DFL leave-one-out folds 1-6, 30 epochs each, ~4.3 h each) -> 300 (DFL + 62 PFF, fold 0, 15 epochs, ~1 day) -> 400 (final model, all 7 DFL + 64 PFF, 15 epochs, ~1 day). ETA for everything: around Sat 2026-10-31 (slipped ~1 h after a stall on Oct 28).
- Check progress: `ssh ... 'ls ~/pathcrf/saved/; grep -c "^Epoch" ~/pathcrf/saved/<trial>/log.txt; tail -2 ~/pathcrf/queue.log; nvidia-smi'`. A trial is finished when `~/pathcrf/saved/<trial>/model/state_dict_best_acc.pt` exists and its log says "Saved Model"; `queue.log` ends with "queue done" when all are done.
- **Copy weights home** (do this for each finished trial; already done for 200 and 201): `scp -i ~/.ssh/vast -P 21901 -r root@ssh4.vast.ai:~/pathcrf/saved/<trial> ~/pathcrf/saved/` and the log `scp ... root@ssh4.vast.ai:~/pathcrf/fold*.log ~/mtah2027/pod/logs/`.
- Score a fold on its held-out German match (fold k holds out the k-th of J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9): `cd ~/mtah2027 && ~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial 20k --dfl <match>`.
- Run the final model on BAR-ATH: `~/pathcrf/.venv/bin/python src/pathcrf_run.py --trial 400 --alive /Users/elijahcavan/mtah2027/evaluation/results/seq.csv` (absolute paths; writes `evaluation/results/pathcrf.csv` and prints the public-window sanity score).
- Publish weights: `cd ~/pathcrf/saved && for t in 200 201 202 203 204 205 206 300 400; do tar czf /tmp/$t.tgz $t; done; gh auth switch -u ecavan; gh release create pathcrf-weights -R JustSomeGuysTM/more-than-a-hack-2027-solution /tmp/*.tgz --title "PathCRF weights" --notes "trained on DFL open data (+PFF WC22 for 300/400), see eda/NOTES.md"`.
- If a stage crashed: `ssh ...`, look at `~/pathcrf/fold*.log` / `full.log`, then relaunch from that stage with `cd ~/pathcrf && nohup bash vast.sh folds k ... &` or `nohup bash vast.sh full &` (usage at the top of `~/mtah2027/pod/vast.sh`). If the box itself died: rent another 4090 with the disk slider at 50 GB, `bash pod/vast.sh pack` on the laptop, scp the tarball, `tar xzf`, `find ~ -name '._*' -delete`, `bash ~/mtah2027/pod/vast.sh setup`, relaunch.
- **When everything is home and uploaded: destroy the instance on the vast.ai page (trash icon)** so it stops billing.
- Only public data (DFL, PFF, PathCRF code) ever went to the box; BAR-ATH never leaves the laptop.

## The pipeline in one paragraph
Per-frame models on the tactical tracking (HGB `src/v2.py`, BiGRU `src/seq.py`; trained on the 7 DFL open-data matches with the ball and players degraded to our tracker's measured quality) -> Viterbi decode into spells -> PathCRF (published possession-path model, player movement only) gives home/away on alive frames, our BiGRU gives the dead balls. Submission candidate file: `evaluation/results/pathcrf.csv` (PathCRF + BiGRU dead balls; public window 0.605 / 0.875, sanity only). Everything tried and dropped (audio, commentary, broadcast ball, our own ball detector, PFF for the frame models) is in `eda/NOTES.md` and the tech doc.

## Results so far (German leave-one-out, realistic degradation; public 5 min in brackets, sanity only)
- HGB 0.438 / 0.828 (0.440 / 0.809). BiGRU 0.464 / 0.839 (0.625 / 0.879). PathCRF authors' ckpt + BiGRU dead balls (0.605 / 0.875).
- PathCRF ours: fold 0 0.608 / 0.933, fold 1 0.579 / 0.905 (perfect dead balls, same matches: HGB 0.451 / 0.870 and 0.442 / 0.853). Authors' checkpoint on its own held-out match 0.727 / 0.949.

## What is left (after the GPU run)
1. Folds 2-6, PFF ablation (300) and final (400) land -> fill the TODO(gpu) cells in `submission/TECH_DOC.md`, pick the final candidate (our PathCRF trial 400 + BiGRU dead balls, unless the folds say otherwise).
2. Upload weights as the release (command above); `bash pod/pathcrf_setup.sh` must then work from a clean clone on a laptop (no GPU needed; ~10 min + download).
3. Clean-clone reproduction of the exact CSV, `python evaluation/validate_submission.py <file>`, rename to `<team_name>.csv` / `<team_name>.md`, make the repo public (or a public copy without data), submit the form. Lock by ~Nov 10.
4. Review/merge PR #6 (Elijah decides when).
Optional, in `eda/NOTES.md` "TODO after the GPU run": dedicated alive/dead model (stoppage onsets ~3 s late), ball detector second attempt (SoccerNet Tracking is in the bucket; `src/external.simulate_detector` judges it honestly).

## If the Claude session is gone
Start a new session in `~/mtah2027`, point it at this file and `eda/NOTES.md`, and ask it to continue the vast.ai queue bookkeeping (copy weights, score folds, publish release, destroy box). Memory file for Claude: `~/.claude/projects/-Users-elijahcavan/memory/mtah2027-competition.md`.
