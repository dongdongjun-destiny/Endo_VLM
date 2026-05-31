#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rennc1/miniconda3/bin:$PATH"
REPO="/home/rennc1/Documents/Yidong_code/Endo_VLM"
"$REPO/sync_exvla_from_local.sh"
cd "$REPO"
[ -f /home/rennc1/Documents/Yidong_code/exendovla_environment.yml ] && \
  cp /home/rennc1/Documents/Yidong_code/exendovla_environment.yml exendovla.yml || true
git pull --rebase origin main
git add -A exvla/ README.md
[ -f exendovla.yml ] && git add exendovla.yml || true
git -c user.name="dongdongjun-destiny" \
    -c user.email="dongdongjun-destiny@users.noreply.github.com" \
    commit -m "Sync latest exvla code and README" || echo "Nothing to commit"
echo ">>> push 时 Password 填 GitHub Token (ghp_...)"
git push origin main
