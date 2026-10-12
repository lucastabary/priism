# Install the priism GPU agent on a Windows PC with an NVIDIA card: the agent in WSL2 (Ubuntu), the tray icon
# (pause / resume, work stats, cache limit) started at every logon. Run it in a normal PowerShell window (not
# as administrator), and rerun it any time: it keeps what is already set up.
#
#   irm https://raw.githubusercontent.com/lucastabary/priism/lab/registry/pod/agent/windows/install.ps1 -OutFile $env:TEMP\priism-install.ps1
#   powershell -ExecutionPolicy Bypass -File $env:TEMP\priism-install.ps1
#
# Before: WSL with Ubuntu (`wsl --install -d Ubuntu` in an administrator PowerShell, reboot, open Ubuntu once
# to create its user) and an up-to-date NVIDIA driver on Windows (it gives the GPU to WSL, nothing to install
# inside Ubuntu). The script asks for the cache folder (where jobs keep data, models and results), its size
# limit, and the two keys Lucas sends.
#
# -LocalRepo <folder>: take install.sh and tray.ps1 from that folder (a priism checkout, or a USB drive with
# pod\agent\...) instead of GitHub. install.sh still clones the priism code from GitHub.
# -Uninstall: stop and remove the agent service and the tray icon; the cache folder stays (delete it by hand).
param(
    [string]$Distro = 'Ubuntu',
    [string]$Branch = 'lab/registry',
    [string]$LocalRepo = '',
    [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'
$OutputEncoding = New-Object System.Text.UTF8Encoding $false  # text piped to WSL
# ... may still arrive with a BOM and CRLF line ends, depending on the console: strip both on the Linux side.
$Clean = 'sed 1s/^\\xEF\\xBB\\xBF// | tr -d \\r '
$Raw = "https://raw.githubusercontent.com/lucastabary/priism/$Branch/pod/agent"
$App = Join-Path $env:LOCALAPPDATA 'priism-agent'
$Shortcut = Join-Path ([Environment]::GetFolderPath('Startup')) 'priism GPU agent.lnk'
$PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$Inv = [Globalization.CultureInfo]::InvariantCulture

function Step([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host "`n$m" -ForegroundColor Red; exit 1 }
function Wsl {
    # Runs a command in the distro (no shell unless asked) and returns its output; fails loudly.
    $out = & wsl.exe -d $Distro @args
    if ($LASTEXITCODE -ne 0) { Fail "Failed in WSL: $($args -join ' ')`n$out" }
    $out
}
function WslSh([string]$script, [switch]$Root) {
    # A shell script through stdin: Windows PowerShell mangles native arguments that hold double quotes.
    $user = if ($Root) { @('-u', 'root') } else { @() }
    $out = ("set -e`n" + $script) | & wsl.exe -d $Distro @user --exec sh -c $Clean'| sh -s'
    if ($LASTEXITCODE -ne 0) { Fail "Failed in WSL:`n$script`n$out" }
    $out
}
function Stop-Tray {
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.CommandLine -like '*priism-agent\tray.ps1*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

if ($Uninstall) {
    Step 'Removing the priism agent'
    & wsl.exe -d $Distro --exec bash -c 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user disable --now priism-agent' 2>$null
    Stop-Tray
    Remove-Item -Force -ErrorAction SilentlyContinue $Shortcut
    Write-Host 'Done. The cache folder and ~/priism-agent in Ubuntu are left in place; delete them by hand if you want.'
    exit 0
}

# ------------------------------------------------------------------------------------------ checks

Step 'Checking WSL and the NVIDIA driver'
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Fail "WSL is not installed. In an administrator PowerShell: wsl --install -d $Distro, reboot, then run this again."
}
$distros = (& wsl.exe -l -q) -replace "`0", '' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($distros -notcontains $Distro) {
    Fail "No '$Distro' in WSL. In an administrator PowerShell: wsl --install -d $Distro, reboot, open $Distro once to create its user, then run this again."
}
$user = ((Wsl --exec whoami) | Select-Object -First 1).Trim()
if ($user -eq 'root') { Fail "$Distro's default user is root: open $Distro once and create a normal user." }
& wsl.exe -d $Distro --exec /usr/lib/wsl/lib/nvidia-smi -L
if ($LASTEXITCODE -ne 0) { Fail 'WSL cannot see an NVIDIA GPU: update the NVIDIA driver on Windows, reboot, and run this again.' }

Step 'Enabling systemd in WSL (the agent is a service that restarts on its own)'
$pid1 = ((Wsl --exec ps -p 1 -o comm=) | Select-Object -First 1).Trim()
if ($pid1 -ne 'systemd') {
    WslSh -Root @'
f=/etc/wsl.conf
touch $f
if grep -q '^\[boot\]' $f; then sed -i '/^systemd *=/d; /^\[boot\]/a systemd=true' $f; else printf '\n[boot]\nsystemd=true\n' >> $f; fi
'@ | Out-Null
    & wsl.exe --terminate $Distro | Out-Null
    $pid1 = ((Wsl --exec ps -p 1 -o comm=) | Select-Object -First 1).Trim()
    if ($pid1 -ne 'systemd') { Fail "systemd did not start in $Distro (PID 1 is $pid1). Run 'wsl --update' and try again." }
}

Step 'Installing system packages in Ubuntu (git, ffmpeg, python venv, C compiler)'
# build-essential, python3-dev: some lab dependencies (MSST: diffq, pesq) build C extensions.
WslSh -Root 'apt-get update -q >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -q git ffmpeg unzip curl python3-venv build-essential python3-dev >/dev/null' | Out-Null
# Linger: the user's services run without a login shell (WSL is kept open by the tray, not by a terminal).
Wsl -u root --exec loginctl enable-linger $user | Out-Null

# ------------------------------------------------------------------------------------------ choices

New-Item -ItemType Directory -Force -Path $App | Out-Null
$ConfigPath = Join-Path $App 'config.json'
$old = if (Test-Path $ConfigPath) { [IO.File]::ReadAllText($ConfigPath) | ConvertFrom-Json } else { $null }

Step 'Cache folder'
if ($old) { $default = $old.cache } else {
    $drive = [IO.DriveInfo]::GetDrives() | Where-Object { $_.DriveType -eq 'Fixed' -and $_.IsReady } |
        Sort-Object AvailableFreeSpace -Descending | Select-Object -First 1
    $default = Join-Path $drive.RootDirectory.FullName 'priism-cache'
}
Write-Host 'Jobs keep everything they download or compute here (data, models, results), up to the size limit.'
$cache = Read-Host "Folder [$default]"
if (-not $cache) { $cache = $default }
$cache = [IO.Path]::GetFullPath($cache.Trim().Trim('"'))
if ($cache -match "['""]") { Fail 'The folder path must not contain quotes.' }
New-Item -ItemType Directory -Force -Path (Join-Path $cache '.agent') | Out-Null

$defaultQuota = if ($old -and $old.quota_gb) { [double]$old.quota_gb } else { 50 }
$answer = Read-Host "Size limit in GB [$defaultQuota]"
$quota = if ($answer) { [double]::Parse($answer.Replace(',', '.'), $Inv) } else { $defaultQuota }
$free = ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($cache))).AvailableFreeSpace / 1e9
if ($free -lt $quota) { Write-Host ('Warning: only {0:N0} GB free on that drive.' -f $free) -ForegroundColor Yellow }

$wslCache = ((Wsl --exec wslpath -a $cache) | Select-Object -First 1).Trim()
$cfg = @{ cache = $cache; distro = $Distro; quota_gb = $quota } | ConvertTo-Json
[IO.File]::WriteAllText($ConfigPath, $cfg)
$ControlPath = Join-Path $cache '.agent\control.json'
$paused = $false
if (Test-Path $ControlPath) { try { $paused = [bool]([IO.File]::ReadAllText($ControlPath) | ConvertFrom-Json).paused } catch { } }
[IO.File]::WriteAllText($ControlPath, (@{ paused = $paused; quota_gb = $quota } | ConvertTo-Json -Compress))

Step 'Keys of the job queue'
& wsl.exe -d $Distro --exec sh -c 'grep -q ^AWS_SECRET_ACCESS_KEY=. ~/.config/priism-agent.env 2>/dev/null'
if ($LASTEXITCODE -eq 0) {
    Write-Host 'Already set (rerun after deleting ~/.config/priism-agent.env in Ubuntu to change them).'
} else {
    Write-Host 'Paste the two keys Lucas sent you.'
    $id = (Read-Host 'Access key ID').Trim()
    $secret = (Read-Host 'Secret access key').Trim()
    if (-not $id -or -not $secret) { Fail 'Both keys are needed.' }
    $envText = @(
        '# The two keys come from Lucas (an R2 token limited to the priism-jobs bucket). Keep this file private.',
        'PRIISM_STORE=s3://priism-jobs/queue',
        'PRIISM_STORE_ENDPOINT=https://f098c4b3a591f78c07382e384992dbd5.r2.cloudflarestorage.com',
        'PRIISM_STORE_REGION=auto',
        "AWS_ACCESS_KEY_ID=$id",
        "AWS_SECRET_ACCESS_KEY=$secret"
    ) -join "`n"
    # Through stdin, not the command line.
    $envText | & wsl.exe -d $Distro --exec sh -c ('umask 077; mkdir -p ~/.config; ' + $Clean + '> ~/.config/priism-agent.env')
    if ($LASTEXITCODE -ne 0) { Fail 'Could not write the keys in Ubuntu.' }
}

# ------------------------------------------------------------------------------------------ agent

Step 'Installing the agent in Ubuntu (PyTorch with CUDA: a few GB, can take 10 minutes)'
if ($LocalRepo) {
    # Copied to C: first: WSL may not see a USB drive plugged in after it started.
    $local = Join-Path $App 'install.sh'
    Copy-Item -Force (Join-Path $LocalRepo 'pod\agent\install.sh') $local
    $script = ((Wsl --exec wslpath -a $local) | Select-Object -First 1).Trim()
} else {
    $script = '/tmp/priism-agent-install.sh'
    Wsl --exec curl -fsSL "$Raw/install.sh" -o $script | Out-Null
}
$q = $quota.ToString($Inv)
& wsl.exe -d $Distro --exec env "BRANCH=$Branch" "WORKDIR=$wslCache" "QUOTA_GB=$q" bash $script
if ($LASTEXITCODE -ne 0) { Fail 'The agent install failed (see above).' }

# ------------------------------------------------------------------------------------------ tray

Step 'Tray icon, started at every logon'
$tray = Join-Path $App 'tray.ps1'
if ($LocalRepo) { Copy-Item -Force (Join-Path $LocalRepo 'pod\agent\windows\tray.ps1') $tray }
else { Invoke-WebRequest -UseBasicParsing "$Raw/windows/tray.ps1" -OutFile $tray }
$trayArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$tray`""
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut($Shortcut)
$lnk.TargetPath = $PowerShellExe
$lnk.Arguments = $trayArgs
$lnk.WindowStyle = 7  # minimized: no window flashes at logon
$lnk.Description = 'priism GPU agent (pause, stats)'
$lnk.Save()
Stop-Tray  # an older tray (reinstall) gives way to the new one
Start-Process -FilePath $PowerShellExe -ArgumentList $trayArgs -WindowStyle Hidden

Write-Host "`nDone." -ForegroundColor Green
Write-Host 'The priism icon is in the notification area (the ^ arrow at the right of the taskbar; drag it onto the'
Write-Host 'taskbar to keep it visible). Click it to pause, resume, see the hours worked, or change the cache limit.'
Write-Host "Cache: $cache (limit $quota GB). Agent logs, in Ubuntu: journalctl --user -u priism-agent -f"
