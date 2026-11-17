# -*- coding: utf-8 -*-
# 使用人员只双击「一键安装.bat」。
# 第一次：用包里自带的库装好训练环境，在桌面放「TRTForge Web」，然后打开软件。
# 以后再双击：训练环境已经在，就只打开软件。

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try { chcp 65001 | Out-Null } catch { }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Exe = Join-Path $Root "app\TRTForgeWeb.exe"
$Python = Join-Path $Root "train-env\python.exe"
$ConfigPath = Join-Path $Root "config.json"
$InstallScript = Join-Path $Root "install_train_env.ps1"

if (-not (Test-Path -LiteralPath $Exe)) {
    Write-Host "找不到程序：$Exe" -ForegroundColor Red
    Write-Host "请先把整个文件夹解压到电脑上，再双击「一键安装.bat」。不要只拷这一个文件。"
    exit 1
}

function Test-TrainReady {
    if (-not (Test-Path -LiteralPath $Python)) { return $false }
    if (-not (Test-Path -LiteralPath $ConfigPath)) { return $false }
    try {
        $cfg = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $false
    }
    return ([string]$cfg.python) -eq $Python
}

if (-not (Test-TrainReady)) {
    if (-not (Test-Path -LiteralPath $InstallScript)) {
        Write-Host "找不到训练环境安装脚本：$InstallScript" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "第一次安装。正在准备训练环境，大约需要十几分钟。" -ForegroundColor Cyan
    Write-Host "使用安装包里自带的库，不需要再联网下载。请不要关闭这个窗口。"
    Write-Host ""
    # 单独开一个 powershell 进程，才能拿到 install_train_env.ps1 里 exit 的退出码。
    $proc = Start-Process -FilePath "powershell.exe" -Wait -PassThru -NoNewWindow -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $InstallScript
    )
    if ($proc.ExitCode -ne 0) {
        Write-Host ""
        Write-Host "训练环境没有装完（退出码 $($proc.ExitCode)）。软件先不要开。把上面的红字留着发给提供安装包的人。" -ForegroundColor Red
        exit $proc.ExitCode
    }
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "TRTForge Web.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $Exe
$shortcut.WorkingDirectory = $Root
$shortcut.Description = "TRTForge Web 缺陷检测"
$shortcut.Save()

Write-Host ""
Write-Host "桌面已放好「TRTForge Web」。以后直接双击那个图标即可。" -ForegroundColor Green
Write-Host "正在打开软件…"
Start-Process -FilePath $Exe -WorkingDirectory $Root
exit 0
