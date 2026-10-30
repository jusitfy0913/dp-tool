# UTF-8
# 使用 torch 虚拟环境生成 TRTForge metadata.json
#
# 用法 1 —— 设置 TORCH_PY 指向 torch 环境的 python.exe（推荐）:
#   $env:TORCH_PY = "D:\venvs\torch\Scripts\python.exe"
#   .\generate_trtforge_metadata.ps1
#
# 用法 2 —— 若虚拟环境放在仓库根目录的 torch\ 或 .venv\ ，脚本会自动探测。
#
param(
    [Parameter(Mandatory = $false)]
    [string]$WeightsDir = "",

    [Parameter(Mandatory = $false)]
    [string]$CalibDir = "",

    [Parameter(Mandatory = $false)]
    [int]$CalibLimit = 32,

    [Parameter(Mandatory = $false)]
    [string]$TorchPython = ""
)

$ErrorActionPreference = "Stop"
$ScriptRoot = $PSScriptRoot
$SrcRoot = Join-Path $ScriptRoot "src"
$Py = $TorchPython
if (-not $Py) { $Py = $env:TORCH_PY }

# 仓库根目录由脚本位置推导：<repo>/2-python-anomaly/anomalib-main
$RepoRoot = (Resolve-Path (Join-Path $ScriptRoot "..\..")).Path

# 默认权重目录：仓库根下的 results，可用 -WeightsDir 覆盖
if (-not $WeightsDir) {
    $WeightsDir = Join-Path $RepoRoot "results\Patchcore\demo_5\latest\weights\lightning"
}

# 默认标定图片目录：脚本同级的 datasets/bottle/test，可用 -CalibDir 覆盖
if (-not $CalibDir) {
    $CalibDir = Join-Path $ScriptRoot "datasets\bottle\test"
}

if (-not $Py -or -not (Test-Path -LiteralPath $Py)) {
    $candidates = @(
        (Join-Path $RepoRoot "torch\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            $Py = $c
            break
        }
    }
}

if (-not $Py -or -not (Test-Path -LiteralPath $Py)) {
    Write-Host @"
未找到 Python 解释器。

请任选其一：
  1) 设置环境变量（推荐，指向你的 torch 虚拟环境）：
     `$env:TORCH_PY = 'D:\你的路径\torch\Scripts\python.exe'
  2) 把 torch 虚拟环境放在仓库根目录：
     $((Join-Path (Resolve-Path (Join-Path $ScriptRoot '..\..')).Path 'torch\Scripts\python.exe'))

然后重新运行本脚本。
"@ -ForegroundColor Yellow
    exit 1
}

$Env:PYTHONPATH = $SrcRoot
$Script = Join-Path $ScriptRoot "trtforge_metadata.py"

Write-Host "Python: $Py"
Write-Host "PYTHONPATH=$SrcRoot"
Write-Host "weights: $WeightsDir"

$argList = @(
    $Script
    $WeightsDir
    "--calib-dir"
    $CalibDir
    "--calib-limit"
    "$CalibLimit"
)
& $Py @argList
