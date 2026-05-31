#!/usr/bin/env bash
set -euo pipefail
LOCAL="/home/rennc1/Documents/Yidong_code/exvla"
REPO="/home/rennc1/Documents/Yidong_code/Endo_VLM/exvla"
rsync -av --delete --exclude 'wandb/' --exclude '__pycache__/' --exclude '*.pyc' "$LOCAL/" "$REPO/"
DIFF_N=$(diff -qr "$LOCAL" "$REPO" --exclude=wandb --exclude='__pycache__' --exclude='*.pyc' 2>/dev/null | wc -l)
echo "diff count: $DIFF_N"
[ "$DIFF_N" -eq 0 ] && echo "OK: 已与本地 exvla 一致"
