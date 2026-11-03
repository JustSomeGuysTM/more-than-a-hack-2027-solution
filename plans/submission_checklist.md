# Submission checklist — Brandon Hiles

**Owner:** Brandon Hiles. Review: Elijah Cavan (Eli), who checks the clean clone before the form.

1. **Final trial.** 401 unless Will's 402 passes its decision rule (`plans/402_retrain.md`). The default trial in
   `pod/pathcrf_setup.sh` and `src/pathcrf_run.py` must be the submitted one (setup was on 400; fixed in the
   `submission-fixes` PR).
2. **Pinned dependencies.** `requirements.txt` and `requirements-model.txt` are the organisers' files and have no torch or
   pyarrow; the PathCRF venv installs torch and torch_geometric unpinned. The rules ask for pinned dependencies and the
   same file. On the training laptop:
   - `~/mtah2027/.venv/bin/pip freeze > requirements-solution.txt`
   - `~/pathcrf/.venv/bin/pip freeze > pod/requirements-pathcrf.txt`
   - make `pod/pathcrf_setup.sh` install the pinned PathCRF file, point TECH_DOC §8 at `requirements-solution.txt`, and
     record the Python version.
3. **Weights must download for a stranger.** `pathcrf_setup.sh` fetches weights with `gh release download` from the team
   repo, which is private and needs a GitHub login. The organisers must be able to fetch them: put the release on the
   public repo and download with plain `curl` from the release URL.
4. **Clean-clone reproduction.** Fresh clone in a new folder, data bundle unpacked, follow TECH_DOC §8 to the letter,
   then `cmp` the result against the CSV being submitted. It must be byte-identical. If it is not, diff the labels to see
   where they differ before changing anything.
5. **Validate**: `python evaluation/validate_submission.py <team_name>.csv` on the exact file being uploaded.
6. **Public repo.**
   - History was scanned on 2026-11-01: no match data. But `src/ball_yolov8n.pt` (trained on our match's frames) stays in
     the history even after the PR removes it, so publish a fresh-history copy rather than flipping this repo public.
   - Leave out or scrub `pod/HANDOVER.md` (box SSH details, account names) and `plans/`.
   - No `data/`, `data_raw/` or frame-level exports.
7. **Tech doc.** Team name. Say 401 was chosen on the DFL validation match (J03WR9, which is also its checkpoint-selection
   match, so the number is optimistic) and that the public window only agreed; that keeps "nothing was tuned on the
   public window" true. Add SoccerNet to the training-data table if the detector is used. Name the files
   `<team_name>.csv` and `<team_name>.md` or `.pdf`.
8. **Form.** Team name, CSV, public repo URL, tech doc. One response per team; the last one counts.
