#!/usr/bin/env bash
# The Docker image's Python environment without Docker: one Python 3.11 virtualenv in .venv with the same pinned packages
# as docker/Dockerfile (CPU torch), and the PathCRF code in ~/pathcrf at the pinned commit with our patch, using that venv.
# Afterwards `bash pod/pathcrf_setup.sh` runs exactly as it does in the container. macOS or Linux (on Windows, use WSL).
#   bash pod/venv_setup.sh                                        # python3.11 on PATH
#   PYTHON=/opt/homebrew/bin/python3.11 bash pod/venv_setup.sh    # or name it
# Needs: Python 3.11, git, internet (about 1 GB of wheels). Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.11}; PC=~/pathcrf; PATHCRF_COMMIT=1b7c6f8; TORCH_VERSION=2.5.1   # as in docker/Dockerfile
"$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null \
  || { echo "Python 3.11 needed (the image runs 3.11.11): install it, or point PYTHON= at one" >&2; exit 1; }

# PathCRF code, as the Dockerfile fetches it (its requirements.txt is part of the install below)
[ -d $PC ] || git clone -q https://github.com/hyunsungkim-ds/pathcrf $PC
git -C $PC checkout -q $PATHCRF_COMMIT
git -C $PC apply --reverse --check "$PWD/pod/pathcrf.patch" 2>/dev/null || git -C $PC apply "$PWD/pod/pathcrf.patch"
mkdir -p $PC/saved

# packages: torch from the CPU wheel index first, then everything else in one resolve, as in the image
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.venv/bin/python -m pip install -q -r requirements-model.txt -r docker/requirements-docker.txt -r $PC/requirements.txt
.venv/bin/python -m pip check

# pod/pathcrf_setup.sh runs PathCRF with ~/pathcrf/.venv/bin/python; make that this venv (in the image both are one Python)
if [ ! -e $PC/.venv ]; then ln -s "$PWD/.venv" $PC/.venv
elif [ "$(cd $PC/.venv && pwd -P)" != "$(cd .venv && pwd -P)" ]; then
  echo "note: $PC/.venv is a separate environment; pod/pathcrf_setup.sh will run PathCRF with it, not with .venv" >&2
fi
.venv/bin/python -c "import sys, torch, numpy, pandas, torch_geometric; print('ready: Python', sys.version.split()[0], '| torch', torch.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__)"
echo "next: bash pod/pathcrf_setup.sh"
