# -*- coding: utf-8 -*-
# 在安装目录旁边创建训练用的 Python 3.12 环境（CUDA 11.8）。
# 只从安装包自带的 wheels 目录安装第三方库，不包含本软件的源码。
#
# 两种模式，脚本自己判断：
# 1. 离线（推荐，安装包默认形态）：安装目录里有 `wheels\` 和 `runtime\python-embed.zip`。
#    用自带的嵌入式 Python 建环境，`pip --no-index --find-links wheels` 装库，
#    整过过程不需要外网。这也是对方电脑没有 Anaconda、又不能联网时的唯一可行路径。
# 2. 联网（老安装包）：找不到 wheels 时按原来的方式装 Miniconda / micromamba，
#    再从 PyPI 下载。只为兼容已经发出去的旧包，新包不应该走到这里。
#
# 虚拟机或没有 AVX2 的 CPU 上，Ultralytics 自带的 polars-runtime-32
# 一导入就会崩溃（退出码 3221225501）。离线包用的是 polars-runtime-compat。

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Prefix = Join-Path $Root "train-env"
$ConfigPath = Join-Path $Root "config.json"
$ToolsDir = Join-Path $Root ".tools"
$WheelsDir = Join-Path $Root "wheels"
$EmbedZip = Join-Path $Root "runtime\python-embed.zip"
#: 与打包脚本里 wheelRequirements 的 polars 版本一致；两边不一致会让
#: `polars-runtime-compat` 装成不匹配的版本，导入时才崩。
$PolarsVersion = "1.36.1"

$HasWheels = (Test-Path -LiteralPath $WheelsDir) -and
    (@(Get-ChildItem -LiteralPath $WheelsDir -Filter *.whl -File -ErrorAction SilentlyContinue).Count -gt 0)
$Offline = $HasWheels -and (Test-Path -LiteralPath $EmbedZip)
$PipOffline = @("--no-index", "--find-links", $WheelsDir)

function Invoke-PipInstall {
    param([string]$Python, [string[]]$Packages, [switch]$Quiet)
    $common = @("-m", "pip", "install", "--timeout", "180")
    if ($Quiet) { $common += "--quiet" }
    if ($HasWheels) {
        & $Python @common @PipOffline @Packages
    } else {
        & $Python @common @Packages
    }
}

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
    if ($HasWheels) {
        Write-Host "从安装包内的 wheels 目录安装 PyTorch（离线，不需要网络）"
        Invoke-PipInstall -Python $Python -Packages @("torch==2.3.0+cu118", "torchvision==0.18.0+cu118")
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "离线安装 PyTorch 失败，改用联网安装。" -ForegroundColor Yellow
    }
    # 联网安装时的三个源。为什么带 +cu118：用 --find-links 时 PyPI 仍在索引里，
    # 不写本地版本号就可能装到 CPU 版 torch（PyPI 上 torch==2.3.0 就是 CPU 版）。
    # 官方 CDN 在部分网络下会返回一个很短的错误页（pip 报哈希不符），所以要有备源；
    # 上海交大实测约 4 MB/s，阿里云只有 0.5 MB/s，慢的那份排最后。
    $sources = @(
        @{ Label = "PyTorch 官方源"; Args = @("--index-url", "https://download.pytorch.org/whl/cu118") },
        @{ Label = "上海交大镜像"; Args = @("--find-links", "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/") },
        @{ Label = "阿里云镜像"; Args = @("--find-links", "https://mirrors.aliyun.com/pytorch-wheels/cu118/") }
    )
    foreach ($source in $sources) {
        Write-Host "安装 PyTorch（CUDA 11.8），源：$($source.Label)"
        $args = @($source.Args) + @(
            "install", "--timeout", "180", "--retries", "3",
            "torch==2.3.0+cu118", "torchvision==0.18.0+cu118"
        )
        & $Python -m pip @args
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "这个源没有装完，换下一个。"
    }
    Write-Host "PyTorch 没有安装成功。请确认能访问 download.pytorch.org / mirror.sjtu.edu.cn / mirrors.aliyun.com 后重新双击。" -ForegroundColor Red
    exit 1
}

function Repair-Polars {
    param([string]$Python)
    $ver = (& $Python -c "import importlib.metadata as m; print(m.version('polars'), end='')").Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ver)) {
        Write-Host "训练库没有带上 Polars，改为单独安装。"
        Invoke-PipInstall -Python $Python -Packages @("polars==$PolarsVersion")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $ver = (& $Python -c "import importlib.metadata as m; print(m.version('polars'), end='')").Trim()
    }
    Write-Host "改用 Polars 兼容包 $ver。没有 AVX2 的 CPU 或虚拟机不会在导入时崩溃。"
    Invoke-PipInstall -Python $Python -Packages @("polars-runtime-compat==$ver")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "没有与 $ver 完全对应的兼容包，改为安装当前可用版本。"
        Invoke-PipInstall -Python $Python -Packages @("polars-runtime-compat")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    # 卸掉需要 AVX2 的运行库。没装过这个包时卸载会返回非 0，不影响后面的步骤。
    & $Python -m pip uninstall -y polars-runtime-32
}

# 用安装包自带的嵌入式 Python 建 train-env，全程不联网。
function Install-EmbeddedPython {
    param([string]$Zip, [string]$Target)
    if (Test-Path -LiteralPath (Join-Path $Target "python.exe")) {
        Write-Host "已有 $Target，继续往里面装库。"
    } else {
        Write-Host "解压自带的 Python 3.12 到 $Target"
        New-Item -ItemType Directory -Force -Path $Target | Out-Null
        Expand-Archive -LiteralPath $Zip -DestinationPath $Target -Force
    }
    $python = Join-Path $Target "python.exe"
    if (-not (Test-Path -LiteralPath $python)) { return $null }
    # 嵌入式发行版默认不允许 import 第三方库：它的 ._pth 只列了 python312.zip 和 .
    # 必须加上 Lib\site-packages 并打开 import site，否则 pip 装完也 import 不到。
    $pth = Join-Path $Target "python312._pth"
    $lines = @("python312.zip", ".", "Lib\site-packages", "import site")
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($pth, $lines, $utf8)
    $sitePackages = Join-Path $Target "Lib\site-packages"
    New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null

    # 嵌入式发行版不带 pip，也没有 ensurepip。pip 本身是纯 Python 的 wheel，
    # 直接解压到 site-packages 就能用——省掉联网跑 get-pip.py。
    & $python -c "import pip" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $pipWheel = Get-ChildItem -LiteralPath $WheelsDir -Filter "pip-*.whl" -File -ErrorAction SilentlyContinue |
            Sort-Object Name | Select-Object -Last 1
        if (-not $pipWheel) { throw "wheels 目录里没有 pip 的 wheel，无法在离线环境里引导 pip。" }
        & $python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" $pipWheel.FullName $sitePackages
        if ($LASTEXITCODE -ne 0) { throw "解压 pip 失败：$($pipWheel.Name)" }
    }
    & $python -m pip --version
    if ($LASTEXITCODE -ne 0) { throw "自带的 Python 里 pip 起不来。" }
    return $python
}

Write-Host "环境将安装到 $Prefix"
if ($Offline) {
    Write-Host "安装包自带 wheels 与嵌入式 Python：全程离线安装，不会联网。"
    $python = Install-EmbeddedPython -Zip $EmbedZip -Target $Prefix
    if (-not $python) {
        Write-Host "自带的 Python 解压失败：$EmbedZip" -ForegroundColor Red
        exit 1
    }
} else {
    if ($HasWheels) {
        Write-Host "找到 wheels 但缺少 runtime\python-embed.zip：库能离线装，Python 环境仍需联网创建。" -ForegroundColor Yellow
    }
    $tool = Ensure-EnvTool
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
}

Install-Torch -Python $python

Write-Host "安装训练库…"
if ($HasWheels) {
    # 一次装齐：AnomalyDINO 的 timm 与 anomalib 共用同一批 torch，分开装会来回解依赖。
    # onnx / onnxruntime / opencv-python 钉在开发机实测过的版本上，理由见打包脚本里同名注释。
    Invoke-PipInstall -Python $python -Packages @(
        "ultralytics==8.4.192", "anomalib==1.2.0", "timm==1.0.14",
        "onnx==1.16.1", "onnxruntime==1.20.1", "opencv-python==4.11.0.86",
        "polars==$PolarsVersion", "polars-runtime-compat==$PolarsVersion"
    )
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    # Ultralytics 与 polars 本身都把 polars-runtime-32 声明成依赖，离线解依赖时
    # wheels 里必须有它，否则 pip 直接报「找不到 polars-runtime-32」。
    # 但它是按「有 AVX2」编的：留在环境里，换到没有 AVX2 的机器上 import 就崩。
    # 所以装完立刻卸掉（那份 wheel 要留着：重跑本脚本时 pip 还要用它解依赖），
    # 最终环境只有 polars-runtime-compat，下面的 import polars 自检就是在验这件事。
    & $python -m pip uninstall -y polars-runtime-32 | Out-Null
    Write-Host "已卸掉 polars-runtime-32（无 AVX2 机器上会崩），环境只用 polars-runtime-compat。"
} else {
    & $python -m pip install --timeout 180 "ultralytics==8.4.192" "anomalib==1.2.0" "opencv-python==4.11.0.86"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Repair-Polars -Python $python

    Write-Host "安装环境自检需要的 onnx 与 onnxruntime…"
    & $python -m pip install --timeout 180 "onnx==1.16.1" "onnxruntime==1.20.1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "安装 AnomalyDINO 需要的主干库 timm…"
    & $python -m pip install --timeout 180 "timm==1.0.14"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "检查训练库能否导入…"
# 逐个 import，而不是写成一长串：一长串里第一个失败就中断，后面缺什么看不见。
# ultralytics / anomalib / cv2 必须在这里验：它们缺了「装完」照样显示成功，
# 用户点「开始训练」才在几秒后报 ModuleNotFoundError，排查时还得让他翻日志。
#
# 为什么写到临时文件再跑、而不是 `python -c "<多行脚本>"`：
# PowerShell 5.1 传原生参数时会把里面的双引号吃掉（实测直接变成 SyntaxError），
# 换成单引号又要避开所有字符串引号，脚本一长就没法读。落一次文件最稳。
# 输出一律 ASCII：这段可能被重定向到日志，中文在非 UTF-8 控制台会抛 UnicodeEncodeError，
# 那样「自检失败」会变成「自检崩了」，反而看不见真正缺的是哪个库。
$selfCheck = @'
import importlib
import sys

mods = ["torch", "timm", "polars", "onnx", "onnxruntime", "cv2", "ultralytics", "anomalib"]
failed = []
for name in mods:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report every missing package
        failed.append(name)
        print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
    else:
        print(("OK   %s %s" % (name, getattr(mod, "__version__", ""))).rstrip())

torch = sys.modules.get("torch")
print("python", sys.version.split()[0])
if torch is not None:
    print("torch", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
if failed:
    print("MISSING: " + ", ".join(failed))
    sys.exit(1)
'@
$selfCheckPath = Join-Path ([System.IO.Path]::GetTempPath()) "trtforge_selfcheck.py"
$selfCheckUtf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($selfCheckPath, $selfCheck, $selfCheckUtf8)
try {
    & $python $selfCheckPath
    $selfCheckCode = $LASTEXITCODE
} finally {
    Remove-Item -LiteralPath $selfCheckPath -ErrorAction SilentlyContinue
}
if ($selfCheckCode -ne 0) {
    Write-Host "导入失败，退出码 $selfCheckCode。若是 3221225501，说明还在使用需要 AVX2 的 Polars 包。" -ForegroundColor Red
    exit $selfCheckCode
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
