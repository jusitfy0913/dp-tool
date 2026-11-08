# -*- coding: utf-8 -*-
# 一键编译两个推理 DLL，不依赖 Qt。
#
# 为什么需要这个脚本：
#   1. OpenCV 路径过去被写死成 D:/thirdparty/opencv，换机器就编不了；
#   2. CUDA 11.8 + MSVC 14.44 这组搭配会触发 STL1002 硬错误，
#      CMake 在探测 CUDA 编译器阶段不会传 CMAKE_CUDA_FLAGS，只能靠环境变量绕。
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_infer_dlls.ps1
#   powershell ... -File scripts\build_infer_dlls.ps1 -OpenCvDir E:\thirdparty\opencv
#   powershell ... -File scripts\build_infer_dlls.ps1 -Targets yolo_infer
#   powershell ... -File scripts\build_infer_dlls.ps1 -OnnxRuntimeRoot E:\thirdparty\onnxruntime
#   powershell ... -File scripts\build_infer_dlls.ps1 -Clean
#
# 产物：<repo>\4-qt-studio\bin\yolo_infer.dll、anomalib_infer.dll、onnx_infer.dll
#
# onnx_infer 是纯 CXX 目标，只需要 ONNX Runtime 头文件与 DLL：
#   先跑 scripts\fetch_onnxruntime.ps1 下载到 3-cpp-tensorrt\onnx_infer\third_party\onnxruntime。
# 缺头文件时它的 CMakeLists 会跳过该目标并打印提示，不影响另外两个库。

param(
    [string]$OpenCvDir = "",
    [string]$CudaDir = "",
    [string]$OnnxRuntimeRoot = "",
    [ValidateSet("all", "yolo_infer", "anomalib_infer", "onnx_infer")]
    [string]$Targets = "all",
    [string]$Configuration = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $RepoRoot "build\infer_dlls"
$BinDir = Join-Path $RepoRoot "4-qt-studio\bin"

# ---------------------------------------------------------------------------
# 定位工具链：优先 PATH，其次常见安装位置
# ---------------------------------------------------------------------------
function Find-CMake {
    $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "E:\3rdlib\CMake\bin\cmake.exe",
        "C:\Program Files\CMake\bin\cmake.exe"
    )) {
        if (Test-Path $p) { return $p }
    }
    throw "找不到 cmake.exe。请安装 CMake 或把它加入 PATH。"
}

function Find-Msvc {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "找不到 vswhere.exe，无法定位 Visual Studio。"
    }
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $install) {
        throw "没有找到装了 C++ 工具集的 Visual Studio 2022。"
    }
    return $install.Trim()
}

# ---------------------------------------------------------------------------
# 解析 OpenCV / CUDA
# ---------------------------------------------------------------------------
if (-not $OpenCvDir) { $OpenCvDir = $env:OPENCV_DIR }
if (-not $OpenCvDir) {
    foreach ($p in @("E:\thirdparty\opencv", "D:\thirdparty\opencv", "C:\thirdparty\opencv")) {
        if (Test-Path (Join-Path $p "OpenCVConfig.cmake")) { $OpenCvDir = $p; break }
    }
}
if (-not $OpenCvDir -or -not (Test-Path (Join-Path $OpenCvDir "OpenCVConfig.cmake"))) {
    throw @"
找不到 OpenCV。请用 -OpenCvDir 指定，或设置环境变量 OPENCV_DIR。
要求目录里有 OpenCVConfig.cmake，例如：
    E:\thirdparty\opencv
"@
}

if (-not $CudaDir) { $CudaDir = $env:CUDA_PATH }
if (-not $CudaDir -and (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")) {
    $CudaDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
}

# ONNX Runtime 头文件目录：不传就用 onnx_infer 子工程里的 third_party 默认值。
# 传了就提前校验，避免编译到一半才报找不到头文件。
if (-not $OnnxRuntimeRoot) { $OnnxRuntimeRoot = $env:ONNXRUNTIME_ROOT }
if ($OnnxRuntimeRoot -and -not (Test-Path (Join-Path $OnnxRuntimeRoot "include\onnxruntime_c_api.h"))) {
    throw "OnnxRuntimeRoot 下找不到 include\onnxruntime_c_api.h：$OnnxRuntimeRoot"
}

$CMake = Find-CMake
$VsInstall = Find-Msvc

# 记录库实际能找到的 OpenCV 版本，避免"编过了但不是这个 OpenCV"的混淆
$opencvVersionFile = Join-Path $OpenCvDir "OpenCVConfig-version.cmake"
$opencvVersion = if (Test-Path $opencvVersionFile) {
    (Select-String -Path $opencvVersionFile -Pattern 'OpenCV_VERSION\s+([0-9.]+)').Matches[0].Groups[1].Value
} else { "未知" }

Write-Host "TRTForge 推理 DLL 编译"
Write-Host "  repo      : $RepoRoot"
Write-Host "  cmake     : $CMake"
Write-Host "  visual std: $VsInstall"
Write-Host "  opencv    : $OpenCvDir (版本 $opencvVersion)"
Write-Host "  cuda      : $(if ($CudaDir) { $CudaDir } else { '(未设置)' })"
Write-Host "  ort       : $(if ($OnnxRuntimeRoot) { $OnnxRuntimeRoot } else { '(用子工程 third_party 默认值)' })"
Write-Host "  targets   : $Targets / $Configuration"
Write-Host ""

# ---------------------------------------------------------------------------
# CUDA 11.8 + MSVC 14.44 的 STL1002 绕过
# 只对需要编译 .cu 的目标有意义，对 CXX-only 目标无副作用。
# ---------------------------------------------------------------------------
$env:NVCC_PREPEND_FLAGS = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH --allow-unsupported-compiler"

if ($Clean -and (Test-Path $BuildRoot)) {
    Write-Host "清理 $BuildRoot"
    Remove-Item $BuildRoot -Recurse -Force
}

$targetsToBuild = switch ($Targets) {
    "all" { @("yolo_infer", "anomalib_infer", "onnx_infer") }
    default { @($Targets) }
}

# 预检：DLL 被占用时链接会报 LNK1104，先给出人能看懂的原因。
# 最常见的占用者就是正在运行的 Web 服务（ctypes 已加载 DLL）或旧的 TRTForge.exe。
foreach ($target in $targetsToBuild) {
    $dll = Join-Path $BinDir "$target.dll"
    if (Test-Path $dll) {
        try {
            $stream = [System.IO.File]::Open($dll, 'Open', 'ReadWrite', 'None')
            $stream.Close()
        } catch {
            Write-Warning @"
$target.dll 正被其他进程占用，无法覆盖。
  原因通常是：Web 服务还在跑（Python 已用 ctypes 加载了 DLL），或 TRTForge.exe 没关。
  处理：先停掉服务（关闭启动窗口，或结束占用该文件的 python 进程），再重跑本脚本。
"@
            throw "目标文件被占用：$dll"
        }
    }
}

$failed = @()

foreach ($target in $targetsToBuild) {
    $sourceDir = Join-Path $RepoRoot "3-cpp-tensorrt\$target"
    $buildDir = Join-Path $BuildRoot $target

    if (-not (Test-Path (Join-Path $sourceDir "CMakeLists.txt"))) {
        Write-Warning "跳过 $target：找不到 $sourceDir\CMakeLists.txt"
        $failed += $target
        continue
    }

    Write-Host "==== 配置 $target ===="
    $configureArgs = @(
        "-S", $sourceDir,
        "-B", $buildDir,
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DOPENCV_DIR=$OpenCvDir",
        "-DTRTFORGE_BIN_DIR=$BinDir"
    )
    if ($CudaDir) { $configureArgs += "-DCUDA_TOOLKIT_ROOT_DIR=$CudaDir" }
    if ($OnnxRuntimeRoot) { $configureArgs += "-DONNXRUNTIME_ROOT=$OnnxRuntimeRoot" }

    & $CMake @configureArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$target 配置失败"
        $failed += $target
        continue
    }

    Write-Host "==== 编译 $target ===="
    & $CMake --build $buildDir --config $Configuration
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$target 编译失败"
        $failed += $target
        continue
    }
}

Write-Host ""
Write-Host "==== 结果 ===="
$produced = @()
foreach ($name in @("yolo_infer", "anomalib_infer", "onnx_infer", "int8_calib")) {
    $dll = Join-Path $BinDir $(if ($name -eq "int8_calib") { "int8_calib.exe" } else { "$name.dll" })
    if (Test-Path $dll) {
        $item = Get-Item $dll
        $produced += [pscustomobject]@{
            Name = $name
            SizeKB = [math]::Round($item.Length / 1KB, 1)
            Updated = $item.LastWriteTime
        }
    }
}
$produced | Format-Table -AutoSize

if ($failed.Count -gt 0) {
    throw "以下目标失败：$($failed -join ', ')"
}

Write-Host "全部成功。产物目录：$BinDir"
Write-Host "提示：运行时如果加载 DLL 失败，检查 OPENCV_DIR 下的 x64\vc17\bin 是否在 DLL 搜索路径。"
