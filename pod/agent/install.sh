#!/usr/bin/env bash
# Install the priism GPU agent on a Linux machine (or WSL2 on a Windows PC with an NVIDIA card), as a
# systemd user service that restarts on its own. Run as the normal user (no root needed except apt).
#
# On a Windows PC, do not run this by hand: pod/agent/windows/install.ps1 sets up WSL, asks for the cache
# folder and its size cap, runs this script inside Ubuntu, and installs the tray icon (pause, stats).
#
# On Linux:
#   curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/lab/registry/pod/agent/install.sh -o install.sh
#   bash install.sh            # then edit ~/.config/priism-agent.env (keys sent by Lucas) and rerun it
#
# What the agent does: it polls a job queue in one Cloudflare R2 bucket, runs allowlisted priism-lab
# commands (no arbitrary scripts) in its work folder (WORKDIR, the size-capped cache: QUOTA_GB), only when
# the GPU is free, and uploads logs and results to that bucket. Its keys give access to that bucket only.
set -euo pipefail
BRANCH=${BRANCH:-lab/registry}
ROOT=$HOME/priism-agent
WORKDIR=${WORKDIR:-$ROOT/work}
QUOTA_GB=${QUOTA_GB:-50}
ENVF=$HOME/.config/priism-agent.env
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}  # systemctl --user from a non-login shell (wsl.exe)
case $WORKDIR in *'"'*|*'\'*) echo "WORKDIR must not contain quotes or backslashes: $WORKDIR"; exit 1;; esac

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver (Windows side for WSL)"; exit 1; }
# build-essential, python3-dev: some lab dependencies (MSST: diffq, pesq) build C extensions.
for pkg in git ffmpeg unzip curl python3-venv build-essential python3-dev; do
  dpkg -s "$pkg" >/dev/null 2>&1 || sudo apt-get install -y -q "$pkg"
done

mkdir -p "$ROOT" "$WORKDIR" "$HOME/.config/systemd/user"
[ -d "$ROOT/priism" ] || git clone https://github.com/lucastabary/priism.git "$ROOT/priism"
git -C "$ROOT/priism" fetch -q origin "$BRANCH"
git -C "$ROOT/priism" checkout -q --detach "origin/$BRANCH"
[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" install -q --upgrade pip
"$ROOT/venv/bin/pip" install -q torch torchaudio boto3 "setuptools<81"  # setuptools: pkg_resources for librosa
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

# PATH: under systemd, WSL's nvidia-smi (/usr/lib/wsl/lib) is not on the default PATH, and without it the
# agent could not see that the owner is using his GPU.
cat > "$HOME/.config/systemd/user/priism-agent.service" <<EOF
[Unit]
Description=priism GPU agent (pulls lab jobs from a queue)
After=network-online.target

[Service]
EnvironmentFile=$ENVF
Environment=PATH=$ROOT/venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib
ExecStart=$ROOT/venv/bin/priism-lab agent --workdir "$WORKDIR" --repo $ROOT/priism --quota-gb $QUOTA_GB
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
systemctl --user enable priism-agent
systemctl --user restart priism-agent
systemctl --user --no-pager status priism-agent | head -5
echo "Logs: journalctl --user -u priism-agent -f"
