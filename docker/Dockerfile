# syntax=docker/dockerfile:1
# One image for the whole possession pipeline: a single Python 3.11 environment with what src/, eda/ and PathCRF need,
# plus an SSH server. Replaces the two virtualenvs (.venv for this repo, ~/pathcrf/.venv for PathCRF).
# The repo is not copied in: docker-compose.yml mounts it at /workspace. See docker/README.md.
#
#   docker compose up -d --build         then         ssh -p 2222 dev@localhost
FROM python:3.11.11-slim-bookworm

# CPU wheels by default; docker/compose.gpu.yml switches TORCH_INDEX to the CUDA ones. PATHCRF_COMMIT matches pod/pathcrf_setup.sh.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1
ARG UID=1000
ARG GID=1000
ARG PATHCRF_COMMIT=1b7c6f8

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ssh, git, gh (pod/pathcrf_setup.sh downloads our PathCRF weights from a GitHub release), ffmpeg (eda/ QA videos), tmux for long runs
RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server git curl ca-certificates build-essential ffmpeg tmux less procps sudo \
 && install -d -m 0755 /etc/apt/keyrings \
 && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

# login user; UID/GID match the host user so files written into the mounted repo stay yours on Linux
RUN groupadd -g ${GID} dev && useradd -m -u ${UID} -g ${GID} -s /bin/bash dev \
 && echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev && chmod 0440 /etc/sudoers.d/dev

# PathCRF code at the pinned commit with our patch, where src/pathcrf_run.py expects it (~/pathcrf)
COPY pod/pathcrf.patch /tmp/pathcrf.patch
USER dev
RUN git clone -q https://github.com/hyunsungkim-ds/pathcrf /home/dev/pathcrf \
 && git -C /home/dev/pathcrf checkout -q ${PATHCRF_COMMIT} \
 && git -C /home/dev/pathcrf apply /tmp/pathcrf.patch \
 && mkdir -p /home/dev/pathcrf/saved /home/dev/pathcrf/data /home/dev/.config/gh
USER root

# python packages: torch from the CPU (or CUDA) wheel index first, then everything else in one resolve
COPY requirements.txt requirements-starter.txt requirements-model.txt docker/requirements-docker.txt /tmp/req/
RUN pip install torch==${TORCH_VERSION} --index-url ${TORCH_INDEX} --extra-index-url https://pypi.org/simple \
 && pip install -r /tmp/req/requirements-model.txt -r /tmp/req/requirements-docker.txt -r /home/dev/pathcrf/requirements.txt \
 && pip check

# floodlight (DFL open data loader) needs numpy 2, while the pipeline keeps the organisers' numpy 1.26 pin, so it gets its
# own Python at /opt/floodlight, used only by `dfl-cache`. It keeps its download (2.4 GB) inside its package folder;
# point that at /data/floodlight, a volume.
RUN python -m venv /opt/floodlight && /opt/floodlight/bin/pip install floodlight==1.2.0 \
 && D=$(/opt/floodlight/bin/python -c "from floodlight.settings import DATA_DIR; print(DATA_DIR)") \
 && rm -rf "$D" && mkdir -p "$(dirname "$D")" /data/floodlight && ln -s /data/floodlight "$D" \
 && chown dev:dev /data/floodlight
COPY --chmod=0755 docker/dfl-cache /usr/local/bin/dfl-cache

# Scripts in the repo call .venv/bin/python and ~/pathcrf/.venv/bin/python; both lead to the one environment here.
# /workspace/.venv is an anonymous volume in docker-compose.yml, so this shim is copied into it and a host .venv stays hidden.
RUN mkdir -p /workspace/.venv/bin /home/dev/pathcrf/.venv/bin \
 && ln -s /usr/local/bin/python /workspace/.venv/bin/python \
 && ln -s /usr/local/bin/python /home/dev/pathcrf/.venv/bin/python \
 && chown -R dev:dev /workspace /home/dev/pathcrf/.venv

# sshd: keys only, user dev only; host keys live in a volume so the fingerprint survives rebuilds
RUN mkdir -p /run/sshd /etc/ssh/host_keys && rm -f /etc/ssh/ssh_host_* \
 && printf '%s\n' \
      'HostKey /etc/ssh/host_keys/ssh_host_ed25519_key' \
      'HostKey /etc/ssh/host_keys/ssh_host_rsa_key' \
      'PasswordAuthentication no' \
      'KbdInteractiveAuthentication no' \
      'PermitRootLogin no' \
      'AllowUsers dev' \
      'X11Forwarding no' \
      > /etc/ssh/sshd_config.d/mtah.conf \
 && echo 'cd /workspace' >> /home/dev/.bashrc
COPY docker/motd /etc/motd
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

WORKDIR /workspace
EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
