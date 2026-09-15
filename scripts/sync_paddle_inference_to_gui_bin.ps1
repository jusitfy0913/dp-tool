# UTF-8
# Sync paddle_inference_install_dir to trtexec/gui/bin (preserve OCR model dirs).

param(
    [string]$InstallDir = "D:\test1\deeplearning\third_party\Paddle\build_infer_trt\paddle_inference_install_dir",
    [string]$DestRoot = "D:\test1\deeplearning\trtexec\gui\bin\paddle_inference",
    [string]$GuiBin = "D:\test1\deeplearning\trtexec\gui\bin"
)

$ErrorActionPreference = "Stop"

function Test-PaddleInstallReady {
    param([string]$Root)
    $dll = Join-Path $Root "paddle\lib\paddle_inference.dll"
    $lib = Join-Path $Root "paddle\lib\paddle_inference.lib"
    return ((Test-Path $dll) -and (Test-Path $lib))
}

if (-not (Test-PaddleInstallReady $InstallDir)) {
    throw "Install dir not ready (missing paddle_inference.dll/lib): $InstallDir"
}

Write-Host "Sync Paddle Inference -> $DestRoot"

$preserveModels = @(
    "ch_PP-OCRv4_det_infer",
    "ch_PP-OCRv4_rec_infer",
    "ch_ppocr_mobile_v2.0_cls_infer",
    "ch_ppocr_mobile_v2.0_cls_slim_infer"
)
$modelBackup = Join-Path $env:TEMP "trtforge_paddle_models_backup"
if (Test-Path $modelBackup) { Remove-Item -Recurse -Force $modelBackup }
New-Item -ItemType Directory -Force -Path $modelBackup | Out-Null

foreach ($name in $preserveModels) {
    $src = Join-Path $DestRoot $name
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $modelBackup $name) -Recurse -Force
        Write-Host "  backup model: $name"
    }
}

# paddle + third_party from install dir
foreach ($sub in @("paddle", "third_party")) {
    $from = Join-Path $InstallDir $sub
    $to = Join-Path $DestRoot $sub
    if (-not (Test-Path $from)) { throw "Missing $from" }
    if (Test-Path $to) { Remove-Item -Recurse -Force $to }
    Copy-Item -Path $from -Destination $to -Recurse -Force
    Write-Host "  copied $sub"
}

foreach ($name in $preserveModels) {
    $bak = Join-Path $modelBackup $name
    if (Test-Path $bak) {
        $dst = Join-Path $DestRoot $name
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Copy-Item -Path $bak -Destination $dst -Recurse -Force
        Write-Host "  restored model: $name"
    }
}

# MKLDNN runtime DLLs to exe dir (TRTForge POST_BUILD also does this)
$mklDir = Join-Path $DestRoot "third_party\install\mklml\lib"
foreach ($dll in @("mklml.dll", "libiomp5md.dll")) {
    $src = Join-Path $mklDir $dll
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $GuiBin $dll) -Force
        Write-Host "  copied to bin: $dll"
    }
}

# optional: copy all DLLs under paddle/lib to bin for easier load
$paddleLib = Join-Path $DestRoot "paddle\lib"
Get-ChildItem $paddleLib -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination (Join-Path $GuiBin $_.Name) -Force
    Write-Host "  copied to bin: $($_.Name)"
}

Write-Host "Sync done."
