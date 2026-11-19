# setup_robot_network.ps1 — 실물 UR5e 를 이 PC (WSL Ubuntu-24.04) 에 연결하는 한 번 설정. 관리자 PowerShell 에서:
#   powershell -ExecutionPolicy Bypass -File ros2_ws\tools\setup_robot_network.ps1
#
# 왜: UR 드라이버는 로봇이 PC 쪽 포트 50001 ~ 50004 로 되접속한다. WSL 기본 (NAT) 은 이 접속을 못 받는다
#     -> WSL 을 mirrored 네트워크 (Windows 와 같은 IP) 로 바꾸고, 그 포트를 방화벽에서 연다.
# 하는 일:
#   1) 유선 '이더넷' (Intel I226-V) 에 고정 IP 192.168.1.10/24 (로봇은 펜던트에서 192.168.1.102/24 로)
#   2) %USERPROFILE%\.wslconfig 에 networkingMode=mirrored (기존 파일은 .bak 으로 남김) — 이 PC 의 모든 WSL 배포판에 적용
#   3) 방화벽: WSL (Hyper-V) 와 Windows 둘 다 TCP 50001-50004 받기 허용
#   4) wsl --shutdown (다음에 WSL 을 열면 새 설정)
# 되돌리기: .wslconfig.bak 을 되돌리고 wsl --shutdown, 이더넷은 '자동 (DHCP)' 로.
param(
    [string]$Adapter = "이더넷",
    [string]$PcIp = "192.168.1.10",
    [int]$Prefix = 24
)
$ErrorActionPreference = "Stop"

Write-Host "1) $Adapter 고정 IP $PcIp/$Prefix"
$a = Get-NetAdapter -Name $Adapter
Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
Set-NetIPInterface -InterfaceIndex $a.ifIndex -Dhcp Disabled
New-NetIPAddress -InterfaceIndex $a.ifIndex -IPAddress $PcIp -PrefixLength $Prefix | Out-Null

Write-Host "2) .wslconfig -> networkingMode=mirrored"
$cfg = Join-Path $env:USERPROFILE ".wslconfig"
if (Test-Path $cfg) { Copy-Item $cfg "$cfg.bak" -Force; $text = Get-Content $cfg -Raw } else { $text = "" }
if ($text -notmatch "networkingMode") {
    if ($text -notmatch "\[wsl2\]") { $text = "[wsl2]`r`n" + $text }
    $text = $text -replace "\[wsl2\]", "[wsl2]`r`nnetworkingMode=mirrored"
    Set-Content -Path $cfg -Value $text -Encoding ascii   # BOM 없이 (WSL 이 읽게)
}
Get-Content $cfg

Write-Host "3) 방화벽 TCP 50001-50004"
$wslVm = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
if (-not (Get-NetFirewallHyperVRule -Name "UR-driver-WSL" -ErrorAction SilentlyContinue)) {
    New-NetFirewallHyperVRule -Name "UR-driver-WSL" -DisplayName "UR driver reverse ports (WSL)" -Direction Inbound `
        -VMCreatorId $wslVm -Protocol TCP -LocalPorts 50001-50004 -Action Allow | Out-Null
}
if (-not (Get-NetFirewallRule -Name "UR-driver" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name "UR-driver" -DisplayName "UR driver reverse ports" -Direction Inbound -Protocol TCP `
        -LocalPort 50001-50004 -Action Allow | Out-Null
}

Write-Host "4) wsl --shutdown"
wsl.exe --shutdown
Write-Host "끝. 로봇 랜선을 '$Adapter' 에 꽂고, WSL 에서: ping 192.168.1.102"
