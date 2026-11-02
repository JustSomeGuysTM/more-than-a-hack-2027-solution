#!/usr/bin/env bash
# Overnight QA renders on public data (DFL folds, PFF), uploaded for the team. BAR-ATH renders are never uploaded:
# the sync below includes only clips/<error type>_*, clips/pff_*, timeline_*.png, pff_*.png and errors.csv.
#   bash eda/qa_batch.sh            # after `eda/qa.py dfl`; ~1-2 h on a laptop
#   aws s3 sync s3://more-than-a-hack-2027/qa/ eda/out/qa/ --profile mtah    # teammates: pull the clips
set -euo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python
for t in missed_change wrong_team_spell spurious_change missed_stoppage dead_onset_early dead_onset_late spurious_stoppage restart_late restart_early; do
  $PY eda/qa.py clips --type $t --n ${N:-10} 2>&1 | grep -c wrote | sed "s/^/$t clips: /"
done
for g in $(ls data_raw/external/pff_labels2_*.npy | sed 's/.*pff_labels2_//; s/\.npy//' | head -${GAMES:-10}); do
  $PY eda/qa.py pff $g --clips 3 > /dev/null 2>&1 && echo "pff $g done"
done
aws s3 sync eda/out/qa/ s3://more-than-a-hack-2027/qa/ --profile mtah --exclude "*" \
  --include "clips/missed_*" --include "clips/wrong_*" --include "clips/spurious_*" --include "clips/dead_*" --include "clips/restart_*" \
  --include "clips/pff_*" --include "timeline_J03*.png" --include "pff_*.png" --include "errors.csv" --exclude "*barath*"
echo "qa batch done $(date)"
