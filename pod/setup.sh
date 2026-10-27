#!/usr/bin/env bash
# One-shot setup for a CPU dev pod (Ubuntu). Run from $HOME:  bash setup.sh
# Needs: GH_TOKEN (private repo), optional AWS creds (profile mtah) for the match data.
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq git ffmpeg build-essential curl >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
git clone https://${GH_TOKEN}@github.com/JustSomeGuysTM/more-than-a-hack-2027-solution.git mtah2027 2>/dev/null || true
git clone https://github.com/hyunsungkim-ds/pathcrf 2>/dev/null || true
cd ~/mtah2027 && git checkout possession-v3 && uv venv -q --python 3.11 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt floodlight "scikit-learn==1.5.2" torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# DFL open data (figshare, 2.4 GB) -> floodlight cache -> our npz caches
.venv/bin/python -c "from floodlight.io.datasets import IDSSEDataset; IDSSEDataset(dataset_dir_name='idsse', match_id='all')"
mkdir -p data_raw/external && .venv/bin/python - <<'PY'
import numpy as np
from src.external import dfl
for m in ["J03WMX","J03WN1","J03WOH","J03WOY","J03WPY","J03WQQ","J03WR9"]:
    d = dfl(m); np.savez_compressed(f"data_raw/external/dfl_{m}.npz", players=d["players"], ball=d["ball"], team=d["team"], labels=d["labels"], period=d["period"]); print(m, d["players"].shape)
PY
# PathCRF env + DFL raw XML linked into its layout + preprocessing
cd ~/pathcrf && uv venv -q --python 3.11 .venv && uv pip install -q --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple numpy pandas tqdm scipy scikit-learn matplotlib rdp pyarrow kloppy torch_geometric
D=$(~/mtah2027/.venv/bin/python -c "from floodlight.settings import DATA_DIR; print(DATA_DIR)")/idsse
mkdir -p data/sportec/metadata data/sportec/event data/sportec/tracking
ln -sf $D/DFL_02_01_matchinformation_*.xml data/sportec/metadata/; ln -sf $D/DFL_03_02_events_raw_*.xml data/sportec/event/; ln -sf $D/DFL_04_03_positions_raw_observed_*.xml data/sportec/tracking/
.venv/bin/python datatools/preprocess.py
echo "setup done"
