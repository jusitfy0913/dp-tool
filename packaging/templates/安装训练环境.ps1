# -*- coding: utf-8 -*-
# 在安装目录旁边创建训练用的 Python 3.12 环境（CUDA 11.8）。
# 只从网上安装第三方库，不包含本软件的源码。
#
# 对方电脑不必预先安装 Anaconda / Miniconda。找不到 conda 时，
# 脚本会下载独立的 micromamba，再在本目录创建 train-env。
# 虚拟机或没有 AVX2 的 CPU 上，Ultralytics 自带的 polars-runtime-32
# 一导入就会崩溃（退出码 3221225501）。这里改用 polars-runtime-compat。

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Prefix = Join-Path $Root "train-env"
$ConfigPath = Join-Path $Root "config.json"
$ToolsDir = Join-Path $Root ".tools"

function Find-Conda {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "D:\Anaconda3\Scripts\conda.exe",
        "C:\Anaconda3\Scripts\conda.exe",
        (Join-Path $Root ".miniconda\Scripts\conda.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Save-Download {
    param([string]$Url, [string]$Dest, [int]$MinBytes)
    $dir = Split-Path -Parent $Dest
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $part = "$Dest.partial"
    if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    try {
        Write-Host "下载 $Url"
        Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing -TimeoutSec 300
        $size = (Get-Item -LiteralPath $part).Length
        if ($size -lt $MinBytes) { throw "文件只有 $size 字节，不像是完整安装包" }
        Move-Item -LiteralPath $part -Destination $Dest -Force
        return $true
    } catch {
        Write-Host "下载失败：$($_.Exception.Message)"
        if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
        return $false
    }
}

function Get-Micromamba {
    $dest = Join-Path $ToolsDir "micromamba.exe"
    if (Test-Path -LiteralPath $dest) { return $dest }
    $urls = @(
        "https://micro.mamba.pm/api/micromamba/win-64/latest",
        "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64.exe"
    )
    foreach ($url in $urls) {
        if (Save-Download -Url $url -Dest $dest -MinBytes 1000000) { return $dest }
    }
    return $null
}

function Install-Miniconda {
    $destRoot = Join-Path $Root ".miniconda"
    $conda = Join-Path $destRoot "Scripts\conda.exe"
    if (Test-Path -LiteralPath $conda) { return $conda }
    if ($destRoot -match " ") {
        Write-Host "安装目录路径里有空格，静默安装 Miniconda 会失败。请把整个文件夹挪到没有空格的路径，例如 D:\TRTForgeWeb，然后重新双击。" -ForegroundColor Red
        return $null
    }
    $installer = Join-Path $ToolsDir "Miniconda3-latest-Windows-x86_64.exe"
    $urls = @(
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe",
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    )
    $got = $false
    foreach ($url in $urls) {
        if (Save-Download -Url $url -Dest $installer -MinBytes 20000000) { $got = $true; break }
    }
    if (-not $got) { return $null }
    Write-Host "正在安装 Miniconda 到 $destRoot（只给当前用户，不改系统 Python）…"
    $arg = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$destRoot"
    $proc = Start-Process -FilePath $installer -ArgumentList $arg -Wait -PassThru
    if ($proc.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $conda)) {
        Write-Host "Miniconda 安装失败，退出码 $($proc.ExitCode)。" -ForegroundColor Red
        return $null
    }
    return $conda
}

function Ensure-EnvTool {
    $conda = Find-Conda
    if ($conda) {
        Write-Host "使用本机已有的 $conda"
        return @{ Kind = "conda"; Exe = $conda }
    }
    Write-Host "本机没有 Anaconda / Miniconda，改为下载独立环境工具。"
    $mm = Get-Micromamba
    if ($mm) { return @{ Kind = "micromamba"; Exe = $mm } }
    Write-Host "独立环境工具下载失败，改为安装 Miniconda。"
    $conda = Install-Miniconda
    if ($conda) { return @{ Kind = "conda"; Exe = $conda } }
    Write-Host "Python 环境工具没有下载成功。请检查网络后重新双击本脚本。" -ForegroundColor Red
    exit 1
}

function New-TrainPrefix {
    param($Tool)
    if ($Tool.Kind -eq "conda") {
        & $Tool.Exe create -y -p $Prefix python=3.12 pip
        return $LASTEXITCODE
    }
    $env:MAMBA_ROOT_PREFIX = Join-Path $Root ".mamba"
    New-Item -ItemType Directory -Force -Path $env:MAMBA_ROOT_PREFIX | Out-Null
    $channels = @(
        "conda-forge",
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
    )
    foreach ($channel in $channels) {
        Write-Host "创建 Python 3.12，频道：$channel"
        & $Tool.Exe create -y -p $Prefix -c $channel python=3.12 pip
        if ($LASTEXITCODE -eq 0) { return 0 }
        Write-Host "这个频道没有创建成功，换下一个。"
    }
    return 1
}

function Install-Torch {
    param([string]$Python)
    $indexes = @(
        "https://download.pytorch.org/whl/cu118",
        "https://mirrors.aliyun.com/pytorch-wheels/cu118"
    )
    foreach ($index in $indexes) {
        Write-Host "安装 PyTorch（CUDA 11.8），源：$index"
        & $Python -m pip install --timeout 180 --retries 3 torch==2.3.0 torchvision==0.18.0 --index-url $index
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "这个源没有装完，换下一个。"
    }
    Write-Host "PyTorch 没有安装成功。请确认能访问 download.pytorch.org 或 mirrors.aliyun.com 后重新双击。" -ForegroundColor Red
    exit 1
}

function Repair-Polars {
    param([string]$Python)
    $ver = (& $Python -c "import importlib.metadata as m; print(m.version('polars'), end='')").Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ver)) {
        Write-Host "训练库没有带上 Polars，改为单独安装。"
        & $Python -m pip install polars
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $ver = (& $Python -c "import importlib.metadata as m; print(m.version('polars'), end='')").Trim()
    }
    Write-Host "改用 Polars 兼容包 $ver。没有 AVX2 的 CPU 或虚拟机不会在导入时崩溃。"
    & $Python -m pip install --timeout 180 "polars-runtime-compat==$ver"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "没有与 $ver 完全对应的兼容包，改为安装当前可用版本。"
        & $Python -m pip install --timeout 180 polars-runtime-compat
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    # 卸掉需要 AVX2 的运行库。没装过这个包时卸载会返回非 0，不影响后面的步骤。
    & $Python -m pip uninstall -y polars-runtime-32
}

$tool = Ensure-EnvTool
Write-Host "环境将安装到 $Prefix"
if (Test-Path -LiteralPath $Prefix) {
    Write-Host "该目录已存在，将继续往里面装库。若要重做，先删掉 train-env 文件夹。"
} else {
    $code = New-TrainPrefix -Tool $tool
    if ($code -ne 0) {
        Write-Host "Python 3.12 环境没有创建成功。" -ForegroundColor Red
        exit $code
    }
}

$python = Join-Path $Prefix "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "没有找到 $python" -ForegroundColor Red
    exit 1
}

Write-Host "更新 pip…"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Install-Torch -Python $python

Write-Host "安装训练库…"
& $python -m pip install --timeout 180 "ultralytics==8.4.192" "anomalib==1.2.0"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Repair-Polars -Python $python

Write-Host "安装环境自检需要的 onnx 与 onnxruntime…"
& $python -m pip install --timeout 180 onnx onnxruntime
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "检查 torch、Polars、onnx 能否导入…"
& $python -c "import torch, polars, onnx, onnxruntime, sys; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('polars', polars.__version__); print('onnx', onnx.__version__); print('onnxruntime', onnxruntime.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "导入失败，退出码 $LASTEXITCODE。若是 3221225501，说明还在使用需要 AVX2 的 Polars 包。" -ForegroundColor Red
    exit $LASTEXITCODE
}

if (Test-Path -LiteralPath $ConfigPath) {
    $cfg = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $cfg.python = $python
    $json = $cfg | ConvertTo-Json -Depth 5
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($ConfigPath, $json + "`r`n", $utf8)
    Write-Host "已把 Python 路径写入 config.json"
}

Write-Host ""
Write-Host "训练环境准备好了。接下来双击 start_web.bat 打开软件。" -ForegroundColor Green
