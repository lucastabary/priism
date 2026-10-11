#!/usr/bin/env bash
# Install the priism GPU agent on a Linux machine (or WSL2 on a Windows PC with an NVIDIA card), as a
# systemd user service that restarts on its own. Run as the normal user (no root needed except apt):
#
#   curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/lab/registry/pod/agent/install.sh -o install.sh
#   bash install.sh            # then edit ~/.config/priism-agent.env (keys sent by Lucas) and rerun it
#
# What the agent does: it polls a job queue in one Cloudflare R2 bucket, runs allowlisted priism-lab
# commands (no arbitrary scripts) in ~/priism-agent/work, only when the GPU is free, and uploads logs and
# results to that bucket. Its keys give access to that bucket only.
#
# WSL2 first (in Windows): `wsl --install -d Ubuntu`, then in Ubuntu enable systemd:
#   printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf      and in PowerShell: wsl --shutdown
# WSL stops when nothing keeps it open: create a Windows scheduled task "at log on" that runs
#   wsl.exe -d Ubuntu --exec /bin/sleep infinity
# (hidden). Stopping it: `systemctl --user stop priism-agent` in Ubuntu, or remove the task.
set -euo pipefail
BRANCH=${BRANCH:-lab/registry}
ROOT=$HOME/priism-agent
ENVF=$HOME/.config/priism-agent.env

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver (Windows side for WSL)"; exit 1; }
for pkg in git ffmpeg unzip curl python3-venv; do
  dpkg -s "$pkg" >/dev/null 2>&1 || sudo apt-get install -y -q "$pkg"
done

mkdir -p "$ROOT/work" "$HOME/.config/systemd/user"
[ -d "$ROOT/priism" ] || git clone https://github.com/lucastabary/priism.git "$ROOT/priism"
git -C "$ROOT/priism" fetch -q origin "$BRANCH"
git -C "$ROOT/priism" checkout -q --detach "origin/$BRANCH"
[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" install -q --upgrade pip
"$ROOT/venv/bin/pip" install -q torch torchaudio boto3
"$ROOT/venv/bin/pip" install -q -e "$ROOT/priism[lab]"
"$ROOT/venv/bin/python" -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

if [ ! -f "$ENVF" ]; then
  cat > "$ENVF" <<'EOF'
# The two keys come from Lucas (an R2 token limited to the priism-jobs bucket). Keep this file private.
PRIISM_STORE=s3://priism-jobs/queue
PRIISM_STORE_ENDPOINT=https://f098c4b3a591f78c07382e384992dbd5.r2.cloudflarestorage.com
PRIISM_STORE_REGION=auto
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
EOF
  chmod 600 "$ENVF"
  echo "Fill $ENVF, then run this script again."
  exit 0
fi

cat > "$HOME/.config/systemd/user/priism-agent.service" <<EOF
[Unit]
Description=priism GPU agent (pulls lab jobs from a queue)
After=network-online.target

[Service]
EnvironmentFile=$ENVF
ExecStart=$ROOT/venv/bin/priism-lab agent --workdir $ROOT/work --repo $ROOT/priism
Restart=always
RestartSec=30
# A job runs in its own session: if the agent restarts, the job keeps going and the agent watches it again.
KillMode=process
Nice=5

[Install]
WantedBy=default.target
EOF
loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now priism-agent
systemctl --user --no-pager status priism-agent | head -5
echo "Logs: journalctl --user -u priism-agent -f"
