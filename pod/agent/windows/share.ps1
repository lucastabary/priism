# Share the priism cache folder on the local network, so Lucas can copy big data (MoisesDB, caches) into it
# and results out of it at LAN speed instead of through the R2 bucket (10 GB free, ~4 MB/s here).
# Files only: jobs still go through the R2 queue and its allowlist; nothing depends on this share when the
# PC is away from that network.
#
# Run in an ADMINISTRATOR PowerShell, once:
#   powershell -ExecutionPolicy Bypass -File share.ps1          # create (prints the password to send Lucas)
#   powershell -ExecutionPolicy Bypass -File share.ps1 -Remove  # remove the share and the account
#
# What it does: a local account "priism-lan" (random password, hidden from the logon screen, no other
# rights), a share "priism-cache" of the cache folder (from %LOCALAPPDATA%\priism-agent\config.json) with
# change rights for that account only, and one firewall rule: SMB (TCP 445) from the local subnet, on the
# Private network profile only (never Public networks).
# -Cache <folder>: the cache folder, when this admin prompt is not the user who installed the agent.
param([switch]$Remove, [string]$Cache = '')
$ErrorActionPreference = 'Stop'
$User, $Share = 'priism-lan', 'priism-cache'
$SpecialAccounts = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\SpecialAccounts\UserList'
$RuleName = 'priism cache share (SMB, private network, local subnet)'

$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Run this in an administrator PowerShell (right click > Run as administrator).' -ForegroundColor Red
    exit 1
}

if ($Remove) {
    if (Get-SmbShare -Name $Share -ErrorAction SilentlyContinue) { Remove-SmbShare -Name $Share -Force }
    if (Get-LocalUser -Name $User -ErrorAction SilentlyContinue) { Remove-LocalUser -Name $User }
    Remove-ItemProperty -Path $SpecialAccounts -Name $User -ErrorAction SilentlyContinue
    Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    Write-Host 'Share, account and firewall rule removed.'
    exit 0
}

# The config is the logged-on user's (the installer wrote it); an elevated prompt of the same user sees it.
$config = Join-Path $env:LOCALAPPDATA 'priism-agent\config.json'
if (-not $Cache) {  # (PowerShell names are case-insensitive: $cache below is this same parameter)
    if (-not (Test-Path $config)) { Write-Host "No $config : run the agent installer first, or pass -Cache <folder>." -ForegroundColor Red; exit 1 }
    $Cache = ([IO.File]::ReadAllText($config) | ConvertFrom-Json).cache
}
if (-not (Test-Path -LiteralPath $cache)) { Write-Host "No folder $cache" -ForegroundColor Red; exit 1 }

$profiles = Get-NetConnectionProfile
if ($profiles | Where-Object { $_.NetworkCategory -eq 'Public' }) {
    Write-Host "This network is set to Public: the share will not be reachable." -ForegroundColor Yellow
    $answer = Read-Host "Set it to Private (only do this on your own or a friend's home network)? [y/N]"
    if ($answer -match '^[yYoO]') {
        $profiles | Where-Object { $_.NetworkCategory -eq 'Public' } | Set-NetConnectionProfile -NetworkCategory Private
    }
}

$chars = [char[]]'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$bytes = New-Object byte[] 24
$rng.GetBytes($bytes)
$password = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
$secure = ConvertTo-SecureString $password -AsPlainText -Force

if (Get-LocalUser -Name $User -ErrorAction SilentlyContinue) {
    Set-LocalUser -Name $User -Password $secure  # rerun: new password
} else {
    New-LocalUser -Name $User -Password $secure -PasswordNeverExpires -UserMayNotChangePassword `
        -Description 'priism: network access to the cache folder only' | Out-Null
}
if (-not (Test-Path $SpecialAccounts)) { New-Item -Path $SpecialAccounts -Force | Out-Null }
New-ItemProperty -Path $SpecialAccounts -Name $User -Value 0 -PropertyType DWord -Force | Out-Null  # not on the logon screen

$acl = Get-Acl -LiteralPath $cache
$rule = New-Object Security.AccessControl.FileSystemAccessRule($User, 'Modify', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
$acl.AddAccessRule($rule)
Set-Acl -LiteralPath $cache -AclObject $acl

if (Get-SmbShare -Name $Share -ErrorAction SilentlyContinue) { Remove-SmbShare -Name $Share -Force }
New-SmbShare -Name $Share -Path $cache -ChangeAccess "$env:COMPUTERNAME\$User" | Out-Null

Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort 445 -Profile Private `
    -RemoteAddress LocalSubnet -Action Allow | Out-Null

$ips = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127|169\.254)\.' -and $_.InterfaceAlias -notmatch 'WSL|vEthernet|Loopback' }
Write-Host "`nDone. Send Lucas these lines (the password is shown only now):" -ForegroundColor Green
Write-Host "  share:    \\$($ips[0].IPAddress)\$Share   (other addresses: $(($ips | ForEach-Object IPAddress) -join ', '))"
Write-Host "  user:     $env:COMPUTERNAME\$User"
Write-Host "  password: $password"
