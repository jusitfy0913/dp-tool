# -*- coding: utf-8 -*-
# 排障用。正常启动不再经过本脚本。
#
# 正常启动：双击 app\TRTForgeWeb.exe（或安装目录里的 start_web.bat）。
# 读 config.json、设环境变量、单实例检查、开浏览器都由 backend\launcher.py 完成，
# 失败时会弹中文说明框，并把全过程写进 runtime\logs\launcher.log。
#
# 什么时候才用这个脚本：
#   1. 双击 exe 没反应，想在窗口里看到完整报错；
#   2. 怀疑单实例判断或端口探测有问题，想绕过启动器直接跑 uvicorn。
#
# 用法（在安装目录里打开命令行）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\start_user.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $Root "config.json"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Host "找不到 config.json：$ConfigPath" -ForegroundColor Red
    exit 1
}

# 这里只为打印摘要而读一遍，环境变量由 exe 内部的 launcher 自己设置。
$cfg = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$port = 8060
if ($cfg.port) { $port = [int]$cfg.port }
$hostName = "127.0.0.1"
if ($cfg.host) { $hostName = [string]$cfg.host }
$openHost = $hostName
if ($openHost -eq "0.0.0.0") { $openHost = "127.0.0.1" }

$exe = Join-Path $Root "app\TRTForgeWeb.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "找不到程序：$exe" -ForegroundColor Red
    Write-Host "请让提供安装包的人重新打包，不要从源码目录启动这一份。" -ForegroundColor Yellow
    exit 1
}

$env:TRTFORGE_HOME = $Root
# 让 launcher 跳过互斥量与开浏览器，直接在当前窗口跑 uvicorn。
$env:TRTFORGE_LEGACY_START = "1"

Write-Host "TRTForge Web Studio（排障启动）"
Write-Host "  安装目录 : $Root"
Write-Host "  地址     : http://${openHost}:$port/"
Write-Host "  日志     : $(Join-Path $Root 'runtime\logs\launcher.log')"
Write-Host "  模式     : 已跳过单实例检查与浏览器打开，需要手动访问上面的地址"
Write-Host ""

& $exe
exit $LASTEXITCODE
