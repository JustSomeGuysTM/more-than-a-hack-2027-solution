#!/usr/bin/env bash
# PathCRF on BAR-ATH from a clean clone: fetch + patch the PathCRF code, make its venv, download our trained
# weights from the team repo's GitHub release, build the BAR-ATH parquet, run inference.
#   bash pod/pathcrf_setup.sh              # trial 300 (final model) -> evaluation/results/pathcrf.csv
#   TRIAL=140 bash pod/pathcrf_setup.sh    # authors' checkpoint instead
# Needs: this repo's .venv (pip install -r requirements.txt), data/ unpacked, `gh auth login` with access to the team repo.
set -euo pipefail
cd "$(dirname "$0")/.."
PC=~/pathcrf; TRIAL=${TRIAL:-300}; PY=${PYTHON:-python3}
[ -d $PC ] || git clone -q https://github.com/hyunsungkim-ds/pathcrf $PC
git -C $PC checkout -q 1b7c6f8 && { git -C $PC apply "$PWD/pod/pathcrf.patch" 2>/dev/null || echo "pathcrf patch already applied"; }
[ -x $PC/.venv/bin/python ] || { $PY -m venv $PC/.venv && $PC/.venv/bin/pip install -q -r $PC/requirements.txt torch torch_geometric rdp pyarrow; }
[ -d $PC/saved/$TRIAL ] || { gh release download pathcrf-weights -R JustSomeGuysTM/more-than-a-hack-2027-solution -p "$TRIAL.tgz" -O - | tar xz -C $PC/saved; }
[ -f evaluation/results/v2.csv ] || .venv/bin/python src/v2.py --out evaluation/results/v2.csv      # dead-ball source
[ -f data_raw/pathcrf/BARATH.parquet ] || .venv/bin/python src/pathcrf_io.py
$PC/.venv/bin/python src/pathcrf_run.py --trial $TRIAL --alive evaluation/results/v2.csv
