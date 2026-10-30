# Build Paddle Inference on Windows (GPU + TensorRT) for TRTForge OCR.
# Doc: docs/Paddle_Inference_Windows_GPU_TRT_编译指南.md
#
# 所有路径默认按脚本位置推导到仓库内，不再写死某台机器的绝对路径：
#   <repo>            = 本脚本所在 scripts 目录的上一级
#   <third_party_root> = <repo>\third_party            （可用 -ThirdPartyRoot 或 PADDLE_THIRD_PARTY_ROOT 覆盖）
#   PaddleSrc         = <third_party_root>\Paddle       （可用 -PaddleSrc 或 PADDLE_SRC 覆盖）
#   BuildDir          = <third_party_root>\Paddle\build_infer_trt（可用 -BuildDir 或 PADDLE_BUILD 覆盖）

param(
    [string]$PaddleBranch = $(if ($env:PADDLE_BRANCH) { $env:PADDLE_BRANCH } else { "release/3.3" }),
    [string]$ThirdPartyRoot = $(if ($env:PADDLE_THIRD_PARTY_ROOT) { $env:PADDLE_THIRD_PARTY_ROOT } else { "" }),
    [string]$PaddleSrc = $(if ($env:PADDLE_SRC) { $env:PADDLE_SRC } else { "" }),
    [string]$BuildDir = $(if ($env:PADDLE_BUILD) { $env:PADDLE_BUILD } else { "" }),
    [string]$CudaRoot = $(if ($env:CUDA_ROOT) { $env:CUDA_ROOT } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" }),
    [string]$CudnnRoot = $(if ($env:CUDNN_ROOT) { $env:CUDNN_ROOT } else { "" }),
    [string]$TensorRtRoot = $(if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { $CudaRoot }),
    [string]$PythonExe = $(if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "" }),
    [switch]$ConfigureOnly,
    [switch]$SkipConfigure,
    [int]$Parallel = 4
)

$ErrorActionPreference = "Stop"

# 仓库根目录：scripts 目录的上一级
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $ThirdPartyRoot) { $ThirdPartyRoot = Join-Path $RepoRoot "third_party" }
if (-not $PaddleSrc) { $PaddleSrc = Join-Path $ThirdPartyRoot "Paddle" }
if (-not $BuildDir) { $BuildDir = Join-Path $PaddleSrc "build_infer_trt" }

function Resolve-CudnnRoot {
    param([string]$Hint)
    if ($Hint -and (Test-Path (Join-Path $Hint "include\cudnn.h"))) { return $Hint }
    $parent = Get-ChildItem "D:\" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "cudnn|tensor|Tensor" } | Select-Object -First 1
    if ($parent) {
        $cudnn = Get-ChildItem $parent.FullName -Directory -Filter "cudnn-windows*" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cudnn -and (Test-Path (Join-Path $cudnn.FullName "include\cudnn.h"))) { return $cudnn.FullName }
    }
    return $null
}

function Resolve-Tool {
    param([string]$Name, [string[]]$Fallbacks)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in $Fallbacks) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    return $null
}

# 工具链默认先找 PATH，再回退到常见安装位置，避免写死某台机器
$CMakeExe = Resolve-Tool -Name "cmake.exe" -Fallbacks @("C:\Program Files\CMake\bin\cmake.exe")
$GitExe = Resolve-Tool -Name "git.exe" -Fallbacks @("C:\Program Files\Git\cmd\git.exe")

# VS 环境脚本：先用 vswhere 探测，再回退到 Community 默认位置
$VcVars = $null
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if ($vswhere -and (Test-Path $vswhere)) {
    $vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath) 2>$null
    if ($vsPath) {
        $candidate = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $candidate) { $VcVars = $candidate }
    }
}
if (-not $VcVars) {
    $defaultVcVars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $defaultVcVars) { $VcVars = $defaultVcVars }
}

# Python 解释器：环境变量 PYTHON_EXE > PATH 中的 python.exe
if (-not $PythonExe) {
    $pyCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pyCmd) { $PythonExe = $pyCmd.Source }
}

if (-not $CMakeExe) { throw "cmake not found; install CMake and add cmake.exe to PATH" }
if (-not $GitExe) { throw "git not found; install Git and add git.exe to PATH" }
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    throw "python not found; set PYTHON_EXE or add python.exe to PATH"
}
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "CUDA nvcc not found: $nvcc" }
$nvInfer = Join-Path $TensorRtRoot "include\NvInfer.h"
if (-not (Test-Path $nvInfer)) { throw "TensorRT NvInfer.h not found: $nvInfer" }

if (-not $CudnnRoot) { $CudnnRoot = Resolve-CudnnRoot -Hint $CudnnRoot }
if (-not $CudnnRoot) { throw "cuDNN not found; set CUDNN_ROOT" }

Write-Host "Paddle Inference build (GPU + TensorRT)"
Write-Host "repo=$RepoRoot"
Write-Host "third_party=$ThirdPartyRoot"
Write-Host "branch=$PaddleBranch src=$PaddleSrc build=$BuildDir"
Write-Host "cuda=$CudaRoot cudnn=$CudnnRoot trt=$TensorRtRoot python=$PythonExe"

$env:PATH = "$(Join-Path $CudnnRoot 'bin');$(Join-Path $CudaRoot 'bin');$env:PATH"
$env:CUDA_PATH = $CudaRoot

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install numpy protobuf wheel ninja

if (-not (Test-Path (Join-Path $PaddleSrc ".git"))) {
    if ($SkipConfigure) { throw "Paddle source not found: $PaddleSrc; run without -SkipConfigure first" }
    New-Item -ItemType Directory -Force -Path (Split-Path $PaddleSrc -Parent) | Out-Null
    & $GitExe clone --depth 1 --branch $PaddleBranch https://github.com/PaddlePaddle/Paddle.git $PaddleSrc
}

if (Test-Path $VcVars) {
    cmd /c "`"$VcVars`" && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

$CudnnLib = Join-Path $CudnnRoot "lib\x64"
if (-not (Test-Path $CudnnLib)) { $CudnnLib = Join-Path $CudnnRoot "lib" }

$trtRootFwd = $TensorRtRoot -replace '\\', '/'
$cudaRootFwd = $CudaRoot -replace '\\', '/'
$cudnnRootFwd = $CudnnRoot -replace '\\', '/'
$cudnnLibFwd = $CudnnLib -replace '\\', '/'
$pythonFwd = $PythonExe -replace '\\', '/'

$cmakeArgs = @(
    "-S", $PaddleSrc,
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DWITH_GPU=ON",
    "-DWITH_TENSORRT=ON",
    "-DTENSORRT_ROOT=$trtRootFwd",
    "-DCUDA_TOOLKIT_ROOT_DIR=$cudaRootFwd",
    "-DCUDNN_ROOT=$cudnnRootFwd",
    "-DCUDNN_LIBRARY=$cudnnLibFwd/cudnn.lib",
    "-DCUDNN_INCLUDE_DIR=$cudnnRootFwd/include",
    "-DWITH_UNITY_BUILD=ON",
    "-DWITH_TESTING=OFF",
    "-DON_INFER=ON",
    "-DWITH_PYTHON=OFF",
    "-DWITH_DISTRIBUTE=OFF",
    "-DCUDA_ARCH_NAME=Auto",
    "-DPYTHON_EXECUTABLE=$pythonFwd"
)

if ($SkipConfigure) {
    Write-Host "Skip configure (-SkipConfigure): reuse $BuildDir"
} else {
    Write-Host "Running cmake configure..."
    & $CMakeExe @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
}

if ($ConfigureOnly) {
    Write-Host "Configure-only done."
    Pop-Location
    exit 0
}

Write-Host "Running ninja -j$Parallel (may take hours)..."
ninja -j$Parallel
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

$installDir = Join-Path $BuildDir "paddle_inference_install_dir"
Write-Host "Done. Install dir: $installDir"
Write-Host "Next: powershell -File `"$PSScriptRoot\sync_paddle_inference_to_bin.ps1`""

Pop-Location
