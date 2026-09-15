# Build Paddle Inference on Windows (GPU + TensorRT) for TRTForge OCR.
# Doc: docs/Paddle_Inference_Windows_GPU_TRT_compile_guide.md

param(
    [string]$PaddleBranch = $(if ($env:PADDLE_BRANCH) { $env:PADDLE_BRANCH } else { "release/3.3" }),
    [string]$PaddleSrc = $(if ($env:PADDLE_SRC) { $env:PADDLE_SRC } else { "D:\test1\deeplearning\third_party\Paddle" }),
    [string]$BuildDir = $(if ($env:PADDLE_BUILD) { $env:PADDLE_BUILD } else { "D:\test1\deeplearning\third_party\Paddle\build_infer_trt" }),
    [string]$CudaRoot = $(if ($env:CUDA_ROOT) { $env:CUDA_ROOT } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" }),
    [string]$CudnnRoot = $(if ($env:CUDNN_ROOT) { $env:CUDNN_ROOT } else { "" }),
    [string]$TensorRtRoot = $(if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { $CudaRoot }),
    [string]$PythonExe = $(if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "D:\Anaconda\python.exe" }),
    [switch]$ConfigureOnly,
    [int]$Parallel = 4
)

$ErrorActionPreference = "Stop"

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

$CMakeExe = "C:\Program Files\CMake\bin\cmake.exe"
$GitExe = "C:\Program Files\Git\cmd\git.exe"
$VcVars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $CMakeExe)) { throw "cmake not found: $CMakeExe" }
if (-not (Test-Path $GitExe)) { throw "git not found: $GitExe" }
if (-not (Test-Path $PythonExe)) { throw "python not found: $PythonExe" }
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "CUDA nvcc not found: $nvcc" }
$nvInfer = Join-Path $TensorRtRoot "include\NvInfer.h"
if (-not (Test-Path $nvInfer)) { throw "TensorRT NvInfer.h not found: $nvInfer" }

if (-not $CudnnRoot) { $CudnnRoot = Resolve-CudnnRoot -Hint $CudnnRoot }
if (-not $CudnnRoot) { throw "cuDNN not found; set CUDNN_ROOT" }

Write-Host "Paddle Inference build (GPU + TensorRT)"
Write-Host "branch=$PaddleBranch src=$PaddleSrc build=$BuildDir"
Write-Host "cuda=$CudaRoot cudnn=$CudnnRoot trt=$TensorRtRoot python=$PythonExe"

$env:PATH = "$(Join-Path $CudnnRoot 'bin');$(Join-Path $CudaRoot 'bin');$env:PATH"
$env:CUDA_PATH = $CudaRoot

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install numpy protobuf wheel ninja

if (-not (Test-Path (Join-Path $PaddleSrc ".git"))) {
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

Write-Host "Running cmake configure..."
& $CMakeExe @cmakeArgs
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

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
Write-Host "Copy paddle/lib, paddle/include and runtime DLLs to trtexec/gui/bin/paddle_inference"

Pop-Location
