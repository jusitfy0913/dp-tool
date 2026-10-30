# -*- coding: utf-8 -*-
# 启动 TRTForge Web Studio。
#
# 路径全部按脚本位置推导，不写死机器绝对路径：
#   <repo>   = 本脚本所在 scripts 目录的上一级
#   backend  = <repo>/6-web/backend
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_web_studio.ps1
#   powershell ... -File scripts\run_web_studio.ps1 -Port 8070
#   powershell ... -File scripts\run_web_studio.ps1 -PythonExe D:\Anaconda3\envs\dpenv\python.exe
#
# 可选环境变量：
#   TRTFORGE_PYTHON      训练用解释器（不设则自动探测）
#   OPENCV_DIR           OpenCV 根目录，如 E:\thirdparty\opencv
#   CUDA_PATH            CUDA 根目录
#   TRTFORGE_WEB_PORT    监听端口，默认 8060
#   TRTFORGE_WEB_HOST    监听地址，默认 127.0.0.1

param(
    [int]$Port = 0,
    [string]$HostName = "",
    [string]$PythonExe = "",
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

# 仓库根目录：scripts 目录的上一级
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackendDir = Join-Path $RepoRoot "6-web\backend"

if (-not (Test-Path (Join-Path $BackendDir "app.py"))) {
    throw "找不到后端入口：$BackendDir\app.py"
}

# 解释器：显式参数 > 环境变量 > PATH > 常见 conda 环境
if (-not $PythonExe) { $PythonExe = $env:TRTFORGE_PYTHON }
if (-not $PythonExe) {
    $candidate = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($candidate) { $PythonExe = $candidate.Source }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    foreach ($p in @("D:\Anaconda3\envs\dpenv\python.exe", "D:\Anaconda3\python.exe")) {
        if (Test-Path $p) { $PythonExe = $p; break }
    }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    throw "找不到 Python。请用 -PythonExe 指定，或设置环境变量 TRTFORGE_PYTHON。"
}

if ($Port -le 0) {
    $Port = if ($env:TRTFORGE_WEB_PORT) { [int]$env:TRTFORGE_WEB_PORT } else { 8060 }
}
if (-not $HostName) {
    $HostName = if ($env:TRTFORGE_WEB_HOST) { $env:TRTFORGE_WEB_HOST } else { "127.0.0.1" }
}

# DLL 依赖：Python 3.8+ 不会用 PATH 找依赖 DLL，这里显式告知后端
if (-not $env:OPENCV_DIR -and (Test-Path "E:\thirdparty\opencv")) {
    $env:OPENCV_DIR = "E:\thirdparty\opencv"
}
if (-not $env:CUDA_PATH -and (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")) {
    $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
}

Write-Host "TRTForge Web Studio"
Write-Host "  repo      : $RepoRoot"
Write-Host "  backend   : $BackendDir"
Write-Host "  python    : $PythonExe"
Write-Host "  opencv    : $(if ($env:OPENCV_DIR) { $env:OPENCV_DIR } else { '(未设置)' })"
Write-Host "  cuda      : $(if ($env:CUDA_PATH) { $env:CUDA_PATH } else { '(未设置)' })"
Write-Host "  listen    : http://${HostName}:$Port"
Write-Host ""

$Env:PYTHONIOENCODING = "utf-8"
$Env:PYTHONUNBUFFERED = "1"

$uvicornArgs = @("-m", "uvicorn", "app:app", "--host", $HostName, "--port", "$Port")
if ($Reload) { $uvicornArgs += "--reload" }

Push-Location $BackendDir
try {
    & $PythonExe @uvicornArgs
} finally {
    Pop-Location
}
