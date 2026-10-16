# priism GPU agent: the owner's icon in the Windows notification area (the arrow at the right of the taskbar).
#
# The agent itself runs in WSL (systemd user service, see ../install.sh). This script, started at logon by a
# shortcut in the Startup folder (made by install.ps1):
# - keeps WSL running (a hidden `wsl.exe ... sleep infinity`), so the agent runs whenever he is logged in;
# - lets him pause and resume, and set the cache size limit, by writing <cache>\.agent\control.json, which
#   the agent reads every few seconds (pause freezes the running job at once, resume continues it);
# - shows the agent's state (<cache>\.agent\status.json) and how long it worked in the last 24 h, 7 days,
#   30 days and in all (<cache>\.agent\stats.json: job seconds per hour, pauses excluded);
# - quiet mode, for the night: the agent duty-cycles the running job (2 s running, 3 s frozen by default,
#   about 40 % of the power, quieter fans). The Quiet mode menu picks Auto (the quiet hours, 22:00 to 08:00
#   by default, editable), Always or Never; the icon turns dark blue while a job runs quietly. The tray also
#   writes this PC's UTC offset (utc_offset_min), so the agent's quiet hours follow Windows' clock;
# - keeps the PC from going to sleep while a job runs (not while paused or idle);
# - updates itself: the agent publishes its checkout's tray.ps1 in <cache>\.agent\ (so an `update` job
#   reaches the tray too); when it differs from the running code and parses, it replaces this file and the
#   tray restarts.
# Everything goes through files in the cache folder: no network, no admin rights.
#
# Config: config.json next to this script, {"cache": "D:\\priism-cache", "distro": "Ubuntu", "quota_gb": 50}.

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing, Microsoft.VisualBasic
Add-Type -Namespace Priism -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@

$SelfPath = $MyInvocation.MyCommand.Path
$SelfBytes = [IO.File]::ReadAllBytes($SelfPath)  # the code running now, compared with what the agent publishes
$Here = Split-Path -Parent $SelfPath
$Config = [IO.File]::ReadAllText((Join-Path $Here 'config.json')) | ConvertFrom-Json
$StateDir = Join-Path $Config.cache '.agent'
$ControlPath = Join-Path $StateDir 'control.json'

$Mutex = New-Object System.Threading.Mutex($false, 'Local\priism-agent-tray')
try { $owned = $Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
if (-not $owned) { exit 0 }  # another tray is running

# ------------------------------------------------------------------------------------------ files

function Read-Json($path) {
    try { if (Test-Path -LiteralPath $path) { return [IO.File]::ReadAllText($path) | ConvertFrom-Json } } catch { }
    return $null
}

function Get-Control {
    $c = Read-Json $ControlPath
    $quota = if ($c -and $c.quota_gb) { [double]$c.quota_gb } else { [double]$Config.quota_gb }
    $quiet = if ($c -and @('auto', 'on', 'off') -contains $c.quiet) { [string]$c.quiet } else { 'auto' }
    $from = if ($c -and $c.quiet_from) { [string]$c.quiet_from } else { '22:00' }
    $to = if ($c -and $c.quiet_to) { [string]$c.quiet_to } else { '08:00' }
    $offset = if ($c -and $null -ne $c.utc_offset_min) { [double]$c.utc_offset_min } else { $null }
    [pscustomobject]@{ paused = [bool]($c -and $c.paused); quota_gb = $quota; quiet = $quiet; quiet_from = $from
                       quiet_to = $to; utc_offset_min = $offset }
}

function Set-Control([hashtable]$changes) {
    # Changes some settings and keeps the others (including ones this tray does not know, from a newer agent).
    $all = @{}
    $c = Read-Json $ControlPath
    if ($c) { foreach ($p in $c.PSObject.Properties) { $all[$p.Name] = $p.Value } }
    foreach ($k in $changes.Keys) { $all[$k] = $changes[$k] }
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    $tmp = "$ControlPath.tmp"
    # WriteAllText: UTF-8 without BOM; written then renamed, so the agent never reads half a file.
    [IO.File]::WriteAllText($tmp, ($all | ConvertTo-Json -Compress))
    Move-Item -Force -LiteralPath $tmp -Destination $ControlPath
}

function Sync-UtcOffset {
    # The agent runs in WSL, whose clock may not be in Windows' time zone: it reads the quiet hours with this
    # offset. Rewritten only when it changes (first run, daylight saving time, travel).
    $c = Get-Control
    $now = [TimeZoneInfo]::Local.GetUtcOffset([DateTime]::Now).TotalMinutes
    if ($null -eq $c.utc_offset_min -or [double]$c.utc_offset_min -ne $now) { Set-Control @{ utc_offset_min = $now } }
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

# Adopt the keep-alive of a previous tray (restart after a self-update) rather than piling up another one.
$script:Keep = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" |
    Where-Object { $_.CommandLine -like '*sleep infinity*' } | Select-Object -First 1 |
    ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
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
    quiet   = New-DotIcon ([System.Drawing.Color]::FromArgb(40, 60, 140)) $false
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
$ItemQuiet = Add-Item 'Quiet mode' $true
function Add-SubItem([string]$text) {
    $i = New-Object System.Windows.Forms.ToolStripMenuItem $text
    [void]$ItemQuiet.DropDownItems.Add($i)
    $i
}
$ItemQuietAuto = Add-SubItem 'Auto: quiet hours'
$ItemQuietOn = Add-SubItem 'Always quiet'
$ItemQuietOff = Add-SubItem 'Never quiet'
[void]$ItemQuiet.DropDownItems.Add((New-Object System.Windows.Forms.ToolStripSeparator))
$ItemQuietHours = Add-SubItem 'Set quiet hours...'
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

    # Quiet mode (the agent says whether it applies now): shown on a running job, and in the menu.
    $quietNow = $s -and $s.quiet -and $age -le 240
    if ($running -and $quietNow) {
        $icon, $title = 'quiet', 'Working quietly'
        $detail += ' (quiet mode: runs 2 s of every 5)'
    }
    $hours = "$($c.quiet_from) to $($c.quiet_to)"
    $ItemQuiet.Text = switch ($c.quiet) {
        'on' { 'Quiet mode: always' }
        'off' { 'Quiet mode: never' }
        default { "Quiet mode: auto, $hours" + $(if ($quietNow) { ' (now)' } else { '' }) }
    }
    $ItemQuietAuto.Text = "Auto: quiet from $hours"
    $ItemQuietAuto.Checked = $c.quiet -eq 'auto'
    $ItemQuietOn.Checked = $c.quiet -eq 'on'
    $ItemQuietOff.Checked = $c.quiet -eq 'off'

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

function Update-Self {
    $new = Join-Path $StateDir 'tray.ps1'
    if (-not (Test-Path -LiteralPath $new)) { return }
    $bytes = [IO.File]::ReadAllBytes($new)
    if ([Convert]::ToBase64String($bytes) -eq [Convert]::ToBase64String($SelfBytes)) { return }
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($new, [ref]$null, [ref]$errors)
    if ($errors.Count -gt 0) { return }  # a broken script never replaces a working tray
    [IO.File]::WriteAllBytes($SelfPath, $bytes)
    $Timer.Stop()
    $Tray.Visible = $false
    $Mutex.ReleaseMutex()
    Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -WindowStyle Hidden `
        -ArgumentList "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$SelfPath`""
    [System.Windows.Forms.Application]::Exit()
}

# ------------------------------------------------------------------------------------------ actions

$ItemPause.add_Click({
    $c = Get-Control
    Set-Control @{ paused = (-not $c.paused) }
    Update-Tray
})

$ItemQuietAuto.add_Click({ Set-Control @{ quiet = 'auto' }; Update-Tray })
$ItemQuietOn.add_Click({ Set-Control @{ quiet = 'on' }; Update-Tray })
$ItemQuietOff.add_Click({ Set-Control @{ quiet = 'off' }; Update-Tray })

$ItemQuietHours.add_Click({
    $c = Get-Control
    $answer = [Microsoft.VisualBasic.Interaction]::InputBox(
        "Quiet hours, start-end (24 h clock), e.g. 22:00-08:00.`n`nIn Auto mode, a job runs only part of the " +
        "time during these hours, so the PC stays quieter.", 'priism quiet hours', "$($c.quiet_from)-$($c.quiet_to)")
    if ($answer) {
        $m = [regex]::Match($answer, '^\s*([01]?\d|2[0-3])[:h.]([0-5]\d)\s*-\s*([01]?\d|2[0-3])[:h.]([0-5]\d)\s*$')
        if ($m.Success) {
            $from = '{0:00}:{1}' -f [int]$m.Groups[1].Value, $m.Groups[2].Value
            $to = '{0:00}:{1}' -f [int]$m.Groups[3].Value, $m.Groups[4].Value
            Set-Control @{ quiet_from = $from; quiet_to = $to }
            Update-Tray
        } else {
            [void][System.Windows.Forms.MessageBox]::Show("Not two times like 22:00-08:00: $answer", 'priism')
        }
    }
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
            Set-Control @{ quota_gb = $gb }
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
$script:Ticks = 0
$Timer.add_Tick({
    try { Sync-UtcOffset } catch { }
    try { Update-Tray } catch { }
    $script:Ticks++
    if ($script:Ticks % 12 -eq 0) { try { Update-Self } catch { } }  # every minute
})
$Timer.Start()
try { Sync-UtcOffset } catch { }
try { Update-Tray } catch { }
[System.Windows.Forms.Application]::Run()
$Tray.Dispose()
