#!/usr/bin/env bash
# PathCRF on BAR-ATH from a clean clone: fetch + patch the PathCRF code, make its venv, download our trained
# weights from the team repo's GitHub release, build the BAR-ATH parquet, run inference.
#   bash pod/pathcrf_setup.sh              # trial 401 (submitted model) -> evaluation/results/pathcrf.csv
#   TRIAL=140 bash pod/pathcrf_setup.sh    # authors' checkpoint instead
# Needs: this repo's .venv (pip install -r requirements.txt), data/ unpacked, `gh auth login` with access to the team repo.
set -euo pipefail
cd "$(dirname "$0")/.."
PC=~/pathcrf; TRIAL=${TRIAL:-401}; PY=${PYTHON:-python3}
[ -d $PC ] || git clone -q https://github.com/hyunsungkim-ds/pathcrf $PC
git -C $PC checkout -q 1b7c6f8 && { git -C $PC apply "$PWD/pod/pathcrf.patch" 2>/dev/null || echo "pathcrf patch already applied"; }
[ -x $PC/.venv/bin/python ] || { $PY -m venv $PC/.venv && $PC/.venv/bin/pip install -q -r $PC/requirements.txt torch torch_geometric rdp pyarrow; }
[ -d $PC/saved/$TRIAL ] || { gh release download pathcrf-weights -R JustSomeGuysTM/more-than-a-hack-2027-solution -p "$TRIAL.tgz" -O - | tar xz -C $PC/saved; }
[ -f evaluation/results/seq.csv ] || .venv/bin/python src/seq.py      # dead-ball source (BiGRU, weights committed)
[ -f data_raw/pathcrf/BARATH.parquet ] || .venv/bin/python src/pathcrf_io.py
$PC/.venv/bin/python src/pathcrf_run.py --trial $TRIAL --alive evaluation/results/seq.csv
[ -f eda/out/seq_barath_logp.npy ] || .venv/bin/python src/seq.py   # BiGRU log-probs for the blend
.venv/bin/python src/seq.py --blend --trial $TRIAL   # soft PathCRF team vote (BLEND_W) -> evaluation/results/pathcrf.csv
