# priism GPU agent: the owner's icon in the Windows notification area (the arrow at the right of the taskbar).
#
# The agent itself runs in WSL (systemd user service, see ../install.sh). This script, started at logon by a
# shortcut in the Startup folder (made by install.ps1):
# - keeps WSL running (a hidden `wsl.exe ... sleep infinity`), so the agent runs whenever he is logged in;
# - lets him pause and resume, and set the cache size limit, by writing <cache>\.agent\control.json, which
#   the agent reads every few seconds (pause freezes the running job at once, resume continues it);
# - shows the agent's state (<cache>\.agent\status.json) and how long it worked in the last 24 h, 7 days,
#   30 days and in all (<cache>\.agent\stats.json: job seconds per hour, pauses excluded);
# - keeps the PC from going to sleep while a job runs (not while paused or idle).
# Everything goes through files in the cache folder: no network, no admin rights.
#
# Config: config.json next to this script, {"cache": "D:\\priism-cache", "distro": "Ubuntu", "quota_gb": 50}.

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing, Microsoft.VisualBasic
Add-Type -Namespace Priism -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Config = [IO.File]::ReadAllText((Join-Path $Here 'config.json')) | ConvertFrom-Json
$StateDir = Join-Path $Config.cache '.agent'
$ControlPath = Join-Path $StateDir 'control.json'

$created = $false
$Mutex = New-Object System.Threading.Mutex($true, 'Local\priism-agent-tray', [ref]$created)
if (-not $created) { exit 0 }  # already running

# ------------------------------------------------------------------------------------------ files

function Read-Json($path) {
    try { if (Test-Path -LiteralPath $path) { return [IO.File]::ReadAllText($path) | ConvertFrom-Json } } catch { }
    return $null
}

function Get-Control {
    $c = Read-Json $ControlPath
    $quota = if ($c -and $c.quota_gb) { [double]$c.quota_gb } else { [double]$Config.quota_gb }
    [pscustomobject]@{ paused = [bool]($c -and $c.paused); quota_gb = $quota }
}

function Set-Control([bool]$paused, [double]$quota) {
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    $tmp = "$ControlPath.tmp"
    # WriteAllText: UTF-8 without BOM; written then renamed, so the agent never reads half a file.
    [IO.File]::WriteAllText($tmp, (@{ paused = $paused; quota_gb = $quota } | ConvertTo-Json -Compress))
    Move-Item -Force -LiteralPath $tmp -Destination $ControlPath
}

function Now-Seconds { [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0 }

function Format-Hours([double]$s) {
    $m = [int][math]::Round($s / 60)
    if ($m -lt 60) { return "$m min" }
    return ('{0} h {1:00}' -f [math]::Floor($m / 60), ($m % 60))
}

function Get-Work {
    # Sums of job seconds over the last 24 h, 7 days, 30 days and in all, from the hourly buckets.
    $st = Read-Json (Join-Path $StateDir 'stats.json')
    $now = (Now-Seconds) / 3600
    $w = @{ d1 = 0.0; d7 = 0.0; d30 = 0.0; all = 0.0 }
    if ($st -and $st.hours) {
        foreach ($p in $st.hours.PSObject.Properties) {
            $age = $now - [double]$p.Name - 1  # age of the end of that hour
            $s = [double]$p.Value
            $w.all += $s
            if ($age -lt 24) { $w.d1 += $s }
            if ($age -lt 24 * 7) { $w.d7 += $s }
            if ($age -lt 24 * 30) { $w.d30 += $s }
        }
    }
    $w
}

# ------------------------------------------------------------------------------------------ WSL

$script:Keep = $null
function Ensure-Wsl {
    # A wsl.exe that never ends keeps the distro, and so systemd and the agent, running.
    if ($script:Keep -and -not $script:Keep.HasExited) { return }
    $script:Keep = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $Config.distro, '--exec', '/bin/sleep', 'infinity') `
        -WindowStyle Hidden -PassThru
}

# ------------------------------------------------------------------------------------------ icon

function New-DotIcon([System.Drawing.Color]$color, [bool]$pauseBars) {
    $bmp = New-Object System.Drawing.Bitmap 32, 32
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.FillEllipse((New-Object System.Drawing.SolidBrush $color), 2, 2, 28, 28)
    if ($pauseBars) {
        $g.FillRectangle([System.Drawing.Brushes]::White, 10, 9, 4, 14)
        $g.FillRectangle([System.Drawing.Brushes]::White, 18, 9, 4, 14)
    }
    $g.Dispose()
    [System.Drawing.Icon]::FromHandle($bmp.GetHicon())
}

$Icons = @{
    running = New-DotIcon ([System.Drawing.Color]::FromArgb(46, 160, 67)) $false
    idle    = New-DotIcon ([System.Drawing.Color]::FromArgb(56, 120, 200)) $false
    paused  = New-DotIcon ([System.Drawing.Color]::FromArgb(230, 140, 20)) $true
    problem = New-DotIcon ([System.Drawing.Color]::FromArgb(200, 50, 50)) $false
    off     = New-DotIcon ([System.Drawing.Color]::FromArgb(130, 130, 130)) $false
}

$Tray = New-Object System.Windows.Forms.NotifyIcon
$Tray.Icon = $Icons.off
$Tray.Text = 'priism GPU agent'
$Tray.Visible = $true

$Menu = New-Object System.Windows.Forms.ContextMenuStrip
function Add-Item([string]$text, [bool]$enabled = $false) {
    $i = New-Object System.Windows.Forms.ToolStripMenuItem $text
    $i.Enabled = $enabled
    [void]$Menu.Items.Add($i)
    $i
}
$ItemState = Add-Item 'priism GPU agent'
$ItemState.Font = New-Object System.Drawing.Font($ItemState.Font, [System.Drawing.FontStyle]::Bold)
$ItemDetail = Add-Item ''
[void]$Menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))
$ItemWork1 = Add-Item ''
$ItemWork7 = Add-Item ''
$ItemWork30 = Add-Item ''
$ItemWorkAll = Add-Item ''
[void]$Menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))
$ItemCache = Add-Item ''
$ItemPause = Add-Item 'Pause' $true
$ItemQuota = Add-Item 'Set cache size limit...' $true
$ItemOpen = Add-Item 'Open cache folder' $true
[void]$Menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))
$ItemQuit = Add-Item 'Quit (stops the agent until next logon)' $true
$Tray.ContextMenuStrip = $Menu

# ------------------------------------------------------------------------------------------ refresh

function Update-Tray {
    Ensure-Wsl
    $c = Get-Control
    $s = Read-Json (Join-Path $StateDir 'status.json')
    $now = Now-Seconds
    $age = if ($s) { $now - [double]$s.time } else { 1e9 }
    $running = $false

    if ($age -gt 240) {
        $icon, $title = 'off', 'Agent not responding'
        $detail = if ($s) { 'Last seen ' + (Format-Hours $age) + ' ago (starting, or stopped)' } else { 'Starting...' }
    } elseif ($c.paused) {
        $icon, $title = 'paused', 'Paused'
        $detail = if ($s.state -eq 'running') { 'Pausing the job...' } elseif ($s.job) { "Job $($s.job) frozen, resumes where it stopped" } else { 'No job will start until you resume' }
    } elseif ($s.state -eq 'running') {
        $icon, $title, $running = 'running', 'Working', $true
        $since = if ($s.job_start) { ' for ' + (Format-Hours ($now - [double]$s.job_start)) } else { '' }
        $detail = "Job $($s.job)$since"
    } elseif ($s.state -like 'waiting: cache full*') {
        $icon, $title = 'problem', 'Cache full'
        $detail = 'Waiting for Lucas to free space (or raise the limit)'
    } elseif ($s.state -like 'waiting: GPU busy*') {
        $icon, $title = 'idle', 'Waiting'
        $detail = 'Your GPU is in use: the job waits until it is free'
    } elseif ($s.state -eq 'offline' -or -not $s.online) {
        $icon, $title = 'problem', 'Offline'
        $detail = 'No connection to the job queue; a running job goes on'
    } else {
        $icon, $title = 'idle', 'Idle'
        $detail = 'Waiting for jobs'
    }

    $Tray.Icon = $Icons[$icon]
    $tip = "priism agent: $title"
    $Tray.Text = $tip.Substring(0, [math]::Min(63, $tip.Length))
    $ItemState.Text = "priism GPU agent: $title"
    $ItemDetail.Text = $detail
    $ItemPause.Text = if ($c.paused) { 'Resume' } else { 'Pause' }

    $w = Get-Work
    $ItemWork1.Text = 'Worked in the last 24 h: ' + (Format-Hours $w.d1)
    $ItemWork7.Text = 'Last 7 days: ' + (Format-Hours $w.d7)
    $ItemWork30.Text = 'Last 30 days: ' + (Format-Hours $w.d30)
    $ItemWorkAll.Text = 'In all: ' + (Format-Hours $w.all)
    $used = if ($s -and $s.used_bytes) { [double]$s.used_bytes / 1e9 } else { 0 }
    $ItemCache.Text = ('Cache: {0:N1} GB of {1:N0} GB' -f $used, $c.quota_gb)
    $ItemCache.ToolTipText = $Config.cache

    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED while a job runs; ES_CONTINUOUS alone lets the PC sleep again.
    [void][Priism.Power]::SetThreadExecutionState($(if ($running) { [uint32]'0x80000001' } else { [uint32]'0x80000000' }))
}

# ------------------------------------------------------------------------------------------ actions

$ItemPause.add_Click({
    $c = Get-Control
    Set-Control (-not $c.paused) $c.quota_gb
    Update-Tray
})

$ItemQuota.add_Click({
    $c = Get-Control
    $answer = [Microsoft.VisualBasic.Interaction]::InputBox(
        "Maximum size of the cache folder, in GB.`n`nOver it, the running job stops and Lucas frees space " +
        "before anything else runs.", 'priism cache size limit', [string]$c.quota_gb)
    if ($answer) {
        $gb = 0.0
        if ([double]::TryParse($answer.Replace(',', '.'), [Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture, [ref]$gb) -and $gb -gt 0) {
            Set-Control $c.paused $gb
            Update-Tray
        } else {
            [void][System.Windows.Forms.MessageBox]::Show("Not a number of GB: $answer", 'priism')
        }
    }
})

$ItemOpen.add_Click({ Start-Process explorer.exe $Config.cache })

$ItemQuit.add_Click({
    $r = [System.Windows.Forms.MessageBox]::Show(
        "Quit stops the agent until your next logon (a running job may be lost).`n`n" +
        "To only free your GPU, use Pause instead.`n`nQuit anyway?", 'priism',
        [System.Windows.Forms.MessageBoxButtons]::YesNo, [System.Windows.Forms.MessageBoxIcon]::Warning)
    if ($r -eq [System.Windows.Forms.DialogResult]::Yes) {
        $Timer.Stop()
        if ($script:Keep -and -not $script:Keep.HasExited) { $script:Keep.Kill() }
        [void][Priism.Power]::SetThreadExecutionState([uint32]'0x80000000')
        $Tray.Visible = $false
        [System.Windows.Forms.Application]::Exit()
    }
})

$Tray.add_MouseClick({ param($sender, $e)
    if ($e.Button -eq [System.Windows.Forms.MouseButtons]::Left) {
        # Left click opens the same menu as right click.
        $m = [System.Windows.Forms.NotifyIcon].GetMethod('ShowContextMenu', [Reflection.BindingFlags]'Instance,NonPublic')
        $m.Invoke($Tray, $null)
    }
})

$Timer = New-Object System.Windows.Forms.Timer
$Timer.Interval = 5000
$Timer.add_Tick({ try { Update-Tray } catch { } })
$Timer.Start()
try { Update-Tray } catch { }
[System.Windows.Forms.Application]::Run()
$Tray.Dispose()
