# -*- coding: utf-8 -*-
# 读取同目录 config.json，设置环境后启动网页程序。
# 由 start_web.bat 调用。不要让用户改 Windows 系统环境变量。

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $Root "config.json"
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Host "找不到 config.json：$ConfigPath" -ForegroundColor Red
    exit 1
}

$cfg = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$port = 8060
if ($cfg.port) { $port = [int]$cfg.port }
$hostName = "127.0.0.1"
if ($cfg.host) { $hostName = [string]$cfg.host }

$python = ""
if ($cfg.python) { $python = [string]$cfg.python }
if (-not $python) {
    $bundled = Join-Path $Root "train-env\python.exe"
    if (Test-Path -LiteralPath $bundled) { $python = $bundled }
}

$roots = @()
if ($cfg.data_roots) {
    foreach ($item in @($cfg.data_roots)) {
        if ($item) { $roots += [string]$item }
    }
}

$env:TRTFORGE_HOME = $Root
$env:TRTFORGE_REPO_ROOT = $Root
$env:TRTFORGE_FRONTEND_DIR = Join-Path $Root "frontend"
$env:TRTFORGE_BIN_DIR = Join-Path $Root "bin"
$env:TRTFORGE_YOLO_LAUNCHER_DIR = Join-Path $Root "launchers"
$env:TRTFORGE_ANOMALIB_LAUNCHER_DIR = Join-Path $Root "launchers"
$env:TRTFORGE_MODEL_DIR = Join-Path $Root "models"
$env:TRTFORGE_OCR_MODEL_DIR = Join-Path $Root "models\ocr"
$env:TRTFORGE_TRTEXEC = Join-Path $Root "bin\trtexec.exe"
$env:TRTFORGE_TOOL_EXE = Join-Path $Root "app\TRTForgeWeb.exe"
$env:OPENCV_BIN_DIR = Join-Path $Root "bin"
$env:CUDA_BIN_DIR = Join-Path $Root "bin"
$env:TRTFORGE_WEB_PORT = "$port"
$env:TRTFORGE_WEB_HOST = $hostName
$env:TRTFORGE_BROWSE_ROOTS = ($roots -join [IO.Path]::PathSeparator)
$tcl = Join-Path $Root "app\tcl8.6"
$tk = Join-Path $Root "app\tk8.6"
if (Test-Path -LiteralPath $tcl) { $env:TCL_LIBRARY = $tcl }
if (Test-Path -LiteralPath $tk) { $env:TK_LIBRARY = $tk }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
if ($python) { $env:TRTFORGE_PYTHON = $python }

$env:PATH = (Join-Path $Root "bin") + [IO.Path]::PathSeparator + $env:PATH

$exe = $env:TRTFORGE_TOOL_EXE
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "找不到程序：$exe" -ForegroundColor Red
    Write-Host "请让提供安装包的人重新打包，不要从源码目录启动这一份。" -ForegroundColor Yellow
    exit 1
}

$listenHost = $hostName
if ($listenHost -eq "0.0.0.0") { $listenHost = "127.0.0.1" }
$url = "http://${listenHost}:$port/"

Write-Host "TRTForge Web Studio"
Write-Host "  地址     : $url"
Write-Host "  训练环境 : $(if ($python) { $python } else { '未配置。只能推理，不能训练。请先运行 install_train_env.bat' })"
Write-Host ""

$probe = {
    param($Url)
    for ($i = 0; $i -lt 120; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri ($Url + "api/health") -UseBasicParsing -TimeoutSec 1
            if ($resp.StatusCode -eq 200) {
                Start-Process $Url
                return
            }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
}
Start-Job -ScriptBlock $probe -ArgumentList $url | Out-Null

& $exe
exit $LASTEXITCODE
