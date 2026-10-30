#Requires -Version 5.1
# Build TRTForge installer with Inno Setup. Requires Release TRTForge.exe under 4-qt-studio/bin.
# Usage: .\build_installer.ps1 [-InnoDir 'D:\Inno Setup 6']

[CmdletBinding()]
param(
    [string]$InnoDir = ""
)

$ErrorActionPreference = "Stop"
$installerDir = $PSScriptRoot
$guiRoot = Split-Path -Parent $installerDir
$binDir = Join-Path $guiRoot "bin"
$exePath = Join-Path $binDir "TRTForge.exe"

if (-not (Test-Path -LiteralPath $exePath)) {
    throw "TRTForge.exe not found. Build release first: $exePath"
}

$candidates = @(
    $(if ($InnoDir) { Join-Path $InnoDir "ISCC.exe" }),
    "D:\Inno Setup 6\ISCC.exe",
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
) | Where-Object { $_ }

$iscc = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $iscc) {
    throw "ISCC.exe not found. Install Inno Setup 6 or run: .\build_installer.ps1 -InnoDir 'D:\Inno Setup 6'"
}

$iss = Join-Path $installerDir "TRTForge_setup.iss"
Write-Host "ISCC: $iscc"
Write-Host "ISS:  $iss"
& $iscc $iss
if ($LASTEXITCODE -ne 0) {
    throw "ISCC failed, exit code $LASTEXITCODE"
}

$dist = Join-Path $guiRoot "dist"
Write-Host "Done. Output folder: $dist"
Get-ChildItem -LiteralPath $dist -Filter "TRTForge_Setup_*_x64.exe" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host $_.FullName
}
