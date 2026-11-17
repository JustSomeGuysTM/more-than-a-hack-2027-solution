# Pipeline container

One Docker image with everything the possession pipeline needs, and an SSH server so you can log in and run models
the same way on a laptop, a desktop or a rented box. It replaces both virtualenvs: this repo's `.venv` and PathCRF's
`~/pathcrf/.venv`. Inside there is one Python 3.11 environment, and `python` is it. Without Docker, `bash pod/venv_setup.sh`
builds the same environment as one virtualenv in `.venv` (same pinned versions, CPU torch) and points `~/pathcrf/.venv` at it.

The repo is not copied into the image. It is mounted at `/workspace`, so edits on your machine show up in the
container straight away, and results the container writes (`evaluation/results/`, `eda/out/`, `data_raw/`) land in
your checkout.

## First run

```bash
cat ~/.ssh/id_ed25519.pub >> docker/authorized_keys     # one public key per line; each teammate adds theirs
docker compose up -d --build                             # first build takes a few minutes
ssh -p 2222 dev@localhost
```

On Linux, build with your own user and group ids so files written into the repo stay yours:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build
```

To type `ssh mtah` instead, add this to `~/.ssh/config`. VS Code and Cursor Remote-SSH can use the same entry.

```
Host mtah
  HostName localhost
  Port 2222
  User dev
```

## What lives where

| Path in the container | What | Kept by |
|---|---|---|
| `/workspace` | this repo, including `data/`, `data_raw/`, `eda/out/` | your checkout (bind mount) |
| `/home/dev/pathcrf` | PathCRF code at commit `1b7c6f8` with `pod/pathcrf.patch` applied | the image |
| `/home/dev/pathcrf/saved` | PathCRF checkpoints: the authors' ones, plus ours once downloaded | volume `pathcrf-saved` |
| `/home/dev/pathcrf/data` | PathCRF preprocessed parquets | volume `pathcrf-data` |
| `/data/floodlight` | DFL open data download (2.4 GB) | volume `floodlight-data` |
| `/home/dev/.config/gh` | `gh auth login` | volume `gh-config` |

`.venv/bin/python` and `~/pathcrf/.venv/bin/python` both point to the container's Python, so `pod/pathcrf_setup.sh`,
`eda/qa_batch.sh` and the commands in docstrings work unchanged.

## Running the models

Start anything long inside tmux (`tmux new -s cv`, detach with `Ctrl-b d`, come back with `tmux attach -t cv`). Jobs
keep running after you log out.

```bash
python src/seq.py --cv          # BiGRU leave-one-match-out on DFL (needs data_raw/external/dfl_*.npz)
python src/seq.py --train       # BiGRU on all 7 DFL matches -> src/seq_model.pt
python src/seq.py               # BiGRU on BAR-ATH -> evaluation/results/seq.csv (needs data/ unpacked)
```

An interactive login starts in `/workspace`. A one-off command starts in `/home/dev`, so `cd` first:
`ssh mtah 'cd /workspace && python src/seq.py'`.

**DFL caches.** If another machine already has `data_raw/external/dfl_*.npz`, copy them into your checkout. Otherwise
build them in the container. This downloads 2.4 GB once into the `floodlight-data` volume:

```bash
dfl-cache              # -> data_raw/external/dfl_<match>.npz (the steps from pod/setup.sh)
dfl-cache --pathcrf    # also prepares PathCRF's DFL data, needed for src/pathcrf_run.py --dfl
```

`dfl-cache` uses its own Python at `/opt/floodlight`, because floodlight needs numpy 2 and the pipeline is pinned to
numpy 1.26 (`requirements-model.txt`). You never need to call that Python yourself.

**PathCRF.** Our trained weights come from a GitHub release on the private team repo, so log in once:

```bash
gh auth login
bash pod/pathcrf_setup.sh       # PathCRF trial 401, then the soft team vote -> evaluation/results/pathcrf.csv
```

Weights you already have on another machine can be copied in from the host instead:
`docker cp ~/pathcrf/saved/401 mtah2027:/home/dev/pathcrf/saved/`.

**Jupyter.** Forward the port when you connect, then start it inside:
`ssh -L 8888:localhost:8888 mtah`, then `jupyter notebook --no-browser --port 8888`.

## Day to day

- Stop and start: `docker compose stop`, `docker compose start`. `docker compose down` removes the container but keeps
  the volumes. `docker compose down -v` also deletes the volumes (DFL download, PathCRF weights, gh login).
- New teammate key: append it to `docker/authorized_keys`, then `docker compose restart`.
- To try a package: `sudo pip install <name>`. It is gone after the next rebuild; to keep it, add it to
  `docker/requirements-docker.txt` and run `docker compose up -d --build`.
- The SSH host key lives in a volume, so rebuilding doesn't change it. If you delete the volumes, clear the old
  fingerprint with `ssh-keygen -R "[localhost]:2222"`.

## Notes

- **CPU only on a Mac.** Docker can't reach Apple's GPU, so torch runs on CPU here. That's slower than MPS for
  training, but it's the same device on every machine, which makes it the right place for the clean-clone
  reproduction check. Give Docker Desktop enough memory (Settings → Resources); 8 GB or more is safer for
  `seq.py --cv`, which holds all training windows in memory.
- **Reaching it from other machines.** The port is bound to `127.0.0.1` by default. `SSH_BIND=0.0.0.0 docker compose
  up -d` exposes it on the network; login is by public key only, as user `dev`, never root. `SSH_PORT` changes the port.
- **NVIDIA GPUs.** `docker compose -f docker-compose.yml -f docker/compose.gpu.yml up -d --build` builds with CUDA 12.4
  wheels and passes the GPUs through. Linux x86_64 with the NVIDIA Container Toolkit only. Not tested yet.
