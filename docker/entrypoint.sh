#!/usr/bin/env bash
# Container start: install the public keys from docker/authorized_keys for user dev, make SSH host keys once (kept in a
# volume), then run sshd in the foreground. Long jobs started over SSH keep running after you log out (use tmux or nohup).
set -euo pipefail
KEYS=/run/authorized_keys                       # docker/authorized_keys, mounted read-only by docker-compose.yml
install -d -m 700 -o dev -g dev /home/dev/.ssh
if [ -s "$KEYS" ]; then
  install -m 600 -o dev -g dev "$KEYS" /home/dev/.ssh/authorized_keys
  echo "authorized $(grep -cvE '^[[:space:]]*(#|$)' "$KEYS") key(s) for dev"
else
  echo "docker/authorized_keys is empty: nobody can log in. Add a public key and restart." >&2
fi
for t in ed25519 rsa; do
  f=/etc/ssh/host_keys/ssh_host_${t}_key; [ -f "$f" ] || ssh-keygen -q -t "$t" -f "$f" -N ''
done
exec /usr/sbin/sshd -D -e
