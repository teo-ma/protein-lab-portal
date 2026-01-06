#!/usr/bin/env bash
set -euo pipefail


SERVICE_SRC="$(cd "$(dirname "$0")" && pwd)/dayhoff-gradio.service"
SERVICE_DST="/etc/systemd/system/dayhoff-gradio.service"

sudo mkdir -p /mnt/data/tmp /mnt/data/bioemu_colabfold /mnt/data/azureuser_cache/huggingface
sudo cp "$SERVICE_SRC" "$SERVICE_DST"

sudo systemctl daemon-reload
sudo systemctl enable --now dayhoff-gradio
sudo systemctl restart dayhoff-gradio

echo
echo "Enabled?: $(sudo systemctl is-enabled dayhoff-gradio || true)"
echo "Active?:  $(sudo systemctl is-active dayhoff-gradio || true)"
sudo systemctl status --no-pager dayhoff-gradio

echo
echo "Logs: sudo journalctl -u dayhoff-gradio -n 100 --no-pager"
