# -*- coding: utf-8 -*-
# 打「推理 + 训练」用户包。对方电脑统一为 CUDA 11.8。
#
# 包里有：编译后的网页程序、编译后的训练启动器、推理 DLL、
# CUDA 11.8 / TensorRT / OpenCV 运行库、网页文件、config.json。
# 包里没有：C++ 源码、网页后端 .py、YOLO/Anomalib 源码树、Qt 程序。
#
# 在 dp-tool 目录执行：
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\package_web_user.ps1
#
# 可选：
#   -OutDir D:\dist
#   -SkipZip

param(
    [string]$OutDir = "",
    [switch]$SkipZip,
    [switch]$WithOcr,
    [string]$OcrModelDir = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Find-RepoRoot {
    param([string]$StartDir)
    $dir = $StartDir
    for ($i = 0; $i -lt 8; $i++) {
        if ((Test-Path -LiteralPath (Join-Path $dir "6-web\backend\frozen_entry.py")) -and
            (Test-Path -LiteralPath (Join-Path $dir "4-qt-studio\bin"))) {
            return $dir
        }
        $parent = Split-Path -Parent $dir
        if (-not $parent -or $parent -eq $dir) { break }
        $dir = $parent
    }
    return $null
}

function Find-Python {
    foreach ($candidate in @(
        "D:\Anaconda3\envs\dpenv\python.exe",
        "C:\Anaconda3\envs\dpenv\python.exe"
    )) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "找不到用来编译的 Python。请安装带 fastapi、opencv 的 dpenv（Python 3.12）。"
}

function Copy-Utf8 {
    param([string]$Source, [string]$Dest)
    $text = [System.IO.File]::ReadAllText($Source)
    # PowerShell 5.1 把没有 BOM 的脚本按系统 ANSI 读。中文 .ps1 必须带 BOM，否则安装脚本会解析失败。
    $withBom = $Dest.EndsWith(".ps1")
    $utf8 = New-Object System.Text.UTF8Encoding $withBom
    [System.IO.File]::WriteAllText($Dest, $text, $utf8)
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Find-RepoRoot -StartDir $scriptDir
if (-not $repoRoot) { throw "找不到仓库根目录（需要 6-web 和 4-qt-studio）。" }

$python = Find-Python
$binRoot = Join-Path $repoRoot "4-qt-studio\bin"
$backend = Join-Path $repoRoot "6-web\backend"
$frontend = Join-Path $repoRoot "6-web\frontend"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "dist"
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$dest = Join-Path $OutDir "TRTForgeWeb"
$build = Join-Path $OutDir "_nuitka_build"
if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
if (Test-Path -LiteralPath $build) { Remove-Item -LiteralPath $build -Recurse -Force }
New-Item -ItemType Directory -Path $dest | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dest "bin") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dest "launchers") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dest "models") | Out-Null
New-Item -ItemType Directory -Path $build | Out-Null

Write-Host "[1/6] 复制推理运行库（CUDA 11.8 / TensorRT / ONNX Runtime / OpenCV）"
$dllNames = @(
    "yolo_infer.dll", "anomalib_infer.dll",
    "onnx_infer.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll",
    "cublas64_11.dll", "cublasLt64_11.dll", "cudart64_110.dll",
    "cudnn64_8.dll", "cudnn_adv_infer64_8.dll", "cudnn_cnn_infer64_8.dll", "cudnn_ops_infer64_8.dll",
    "nvinfer.dll", "nvinfer_builder_resource.dll", "nvinfer_plugin.dll", "nvonnxparser.dll",
    "dxcompiler.dll", "dxil.dll", "opengl32sw.dll"
)
foreach ($name in $dllNames) {
    $src = Join-Path $binRoot $name
    if (-not (Test-Path -LiteralPath $src)) { throw "缺少运行库：$src" }
    Copy-Item -LiteralPath $src -Destination (Join-Path $dest "bin\$name") -Force
}
# 加速 provider 是可选件：官方 CPU 包只有 providers_shared，CUDA 包才带 providers_cuda。
# 有就一起带上，没有就让 onnx_infer 退回 CPU，而不是让整个安装包失败。
foreach ($name in @("onnxruntime_providers_cuda.dll", "onnxruntime_providers_tensorrt.dll")) {
    $src = Join-Path $binRoot $name
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $dest "bin\$name") -Force
    } else {
        Write-Host "  跳过可选运行库：$name（ONNX 将使用 CPU）"
    }
}
Get-ChildItem -LiteralPath $binRoot -Filter "opencv_*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $dest "bin\$($_.Name)") -Force
}

if ($WithOcr) {
    Write-Host "[1b] 复制 OCR 运行库（paddle_inference.dll 约 1.7GB）"
    $ocrNames = @(
        "ocr_infer.dll", "paddle_inference.dll", "common.dll",
        "mkldnn.dll", "libiomp5md.dll", "yaml-cpp.dll", "polyclipping.dll"
    )
    foreach ($name in $ocrNames) {
        $src = Join-Path $binRoot $name
        if (-not (Test-Path -LiteralPath $src)) { throw "缺少 OCR 运行库：$src" }
        Copy-Item -LiteralPath $src -Destination (Join-Path $dest "bin\$name") -Force
    }
    $mklCandidates = @(
        (Join-Path $binRoot "mklml.dll"),
        (Join-Path $repoRoot "ocr\ocr_infer\third_party\paddle_inference\third_party\install\mklml\lib\mklml.dll")
    )
    $mkl = $mklCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $mkl) { throw "缺少 mklml.dll，OCR 在 CPU 上无法启动" }
    Copy-Item -LiteralPath $mkl -Destination (Join-Path $dest "bin\mklml.dll") -Force
    $versionSrc = Join-Path $repoRoot "ocr\ocr_infer\third_party\paddle_inference\version.txt"
    if (Test-Path -LiteralPath $versionSrc) {
        Copy-Item -LiteralPath $versionSrc -Destination (Join-Path $dest "bin\paddle_inference_version.txt") -Force
    }
    if ($OcrModelDir -and (Test-Path -LiteralPath $OcrModelDir)) {
        $ocrDest = Join-Path $dest "models\ocr"
        New-Item -ItemType Directory -Path $ocrDest -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $OcrModelDir "*") -Destination $ocrDest -Recurse -Force
    }
}

# 运行库清单自检：现场排查「DLL 到底带上没有」时，先看这张表，不用依赖放文件的人。
$requiredBin = @(
    "yolo_infer.dll", "anomalib_infer.dll", "onnx_infer.dll",
    "onnxruntime.dll", "onnxruntime_providers_shared.dll"
)
$missingBin = $requiredBin | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $dest "bin\$_"))
}
if ($missingBin) { throw "安装包 bin 目录缺少：$($missingBin -join ', ')" }
$manifestLines = Get-ChildItem -LiteralPath (Join-Path $dest "bin") -File |
    Sort-Object Name |
    ForEach-Object {
        "{0}`t{1}`t{2}" -f $_.Name, $_.Length, (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
[System.IO.File]::WriteAllLines(
    (Join-Path $dest "bin_manifest.txt"),
    $manifestLines,
    (New-Object System.Text.UTF8Encoding $false)
)

$trtexecCandidates = @(
    "$env:CUDA_PATH\bin\trtexec.exe",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\trtexec.exe"
)
$trtexec = $trtexecCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $trtexec) { throw "找不到 CUDA 11.8 的 trtexec.exe。请安装 CUDA 11.8，或设置 CUDA_PATH。" }
Copy-Item -LiteralPath $trtexec -Destination (Join-Path $dest "bin\trtexec.exe") -Force

Write-Host "[2/6] 复制网页界面"
& robocopy $frontend (Join-Path $dest "frontend") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "复制前端失败，robocopy 退出码 $LASTEXITCODE" }

Write-Host "[3/6] 编译训练启动器（.pyd，不含 torch 本体）"
& $python -m pip install -q nuitka ordered-set zstandard
if ($LASTEXITCODE -ne 0) { throw "安装 Nuitka 失败" }

$nofollow = @(
    "torch", "torchvision", "ultralytics", "anomalib",
    "lightning", "pytorch_lightning", "cv2", "onnx", "onnxruntime"
)
function Compile-Launcher {
    param([string]$SourcePy, [string]$ModuleName)
    $work = Join-Path $build $ModuleName
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    $moduleFile = Join-Path $work "$ModuleName.py"
    Copy-Item -LiteralPath $SourcePy -Destination $moduleFile -Force
    $nuitkaArgs = @(
        "-m", "nuitka", "--module", $moduleFile,
        "--output-dir=$work",
        "--remove-output",
        "--assume-yes-for-downloads"
    )
    foreach ($name in $nofollow) { $nuitkaArgs += "--nofollow-import-to=$name" }
    & $python @nuitkaArgs
    if ($LASTEXITCODE -ne 0) { throw "编译失败：$SourcePy" }
    $pyd = Get-ChildItem -LiteralPath $work -Filter "$ModuleName*.pyd" | Select-Object -First 1
    if (-not $pyd) { throw "没有生成 $ModuleName.pyd" }
    Copy-Item -LiteralPath $pyd.FullName -Destination (Join-Path $dest "launchers\$($pyd.Name)") -Force
}

Compile-Launcher -SourcePy (Join-Path $repoRoot "1-python-yolo\launcher\tf_yolo_launcher.py") -ModuleName "tf_yolo_launcher_bin"
Compile-Launcher -SourcePy (Join-Path $repoRoot "2-python-anomaly\launcher\tf_anomalib_launcher.py") -ModuleName "tf_anomalib_launcher_bin"

$stub = @"
# -*- coding: utf-8 -*-
from __future__ import annotations

import {0} as impl

raise SystemExit(impl.main())
"@
$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Join-Path $dest "launchers\tf_yolo_launcher.py"), ($stub -f "tf_yolo_launcher_bin"), $utf8)
[System.IO.File]::WriteAllText((Join-Path $dest "launchers\tf_anomalib_launcher.py"), ($stub -f "tf_anomalib_launcher_bin"), $utf8)

Write-Host "[4/6] 编译网页程序（这一步较久）"
$serverOut = Join-Path $build "server"
New-Item -ItemType Directory -Path $serverOut -Force | Out-Null
$serverArgs = @(
    "-m", "nuitka", "--standalone",
    "--output-dir=$serverOut",
    "--output-filename=TRTForgeWeb.exe",
    "--assume-yes-for-downloads",
    "--enable-plugin=tk-inter",
    "--windows-console-mode=force",
    "--include-module=tools.val_engine",
    "--include-module=tools.val_anomaly",
    "--include-module=tools.infer_batch",
    "--include-module=tools.infer_video",
    "--include-module=uvicorn.logging",
    "--include-module=uvicorn.loops",
    "--include-module=uvicorn.loops.auto",
    "--include-module=uvicorn.protocols.http.auto",
    "--include-module=uvicorn.protocols.websockets.auto",
    "--include-module=uvicorn.lifespan.on",
    (Join-Path $backend "frozen_entry.py")
)
foreach ($name in @("torch", "torchvision", "ultralytics", "anomalib", "lightning", "pytorch_lightning")) {
    $serverArgs += "--nofollow-import-to=$name"
}
Push-Location $backend
try {
    & $python @serverArgs
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { throw "编译网页程序失败" }

$distDir = Join-Path $serverOut "frozen_entry.dist"
if (-not (Test-Path -LiteralPath $distDir)) { throw "找不到编译结果：$distDir" }
& robocopy $distDir (Join-Path $dest "app") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "复制编译结果失败" }

# Anaconda 把 SSL 和 Tk 的 DLL 放在 Library\bin，Nuitka 不会自动带走。
$envRoot = Split-Path -Parent $python
$libBin = Join-Path $envRoot "Library\bin"
$appDir = Join-Path $dest "app"
foreach ($name in @(
    "libssl-3-x64.dll", "libcrypto-3-x64.dll", "ffi.dll", "sqlite3.dll", "zlib.dll",
    "tcl86t.dll", "tk86t.dll"
)) {
    $from = Join-Path $libBin $name
    if (Test-Path -LiteralPath $from) {
        Copy-Item -LiteralPath $from -Destination (Join-Path $appDir $name) -Force
    }
}
foreach ($pair in @(
    @{ Src = Join-Path $envRoot "Library\lib\tcl8.6"; Dest = Join-Path $appDir "tcl8.6" },
    @{ Src = Join-Path $envRoot "Library\lib\tk8.6"; Dest = Join-Path $appDir "tk8.6" }
)) {
    if (Test-Path -LiteralPath $pair.Src) {
        & robocopy $pair.Src $pair.Dest /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "复制 $($pair.Src) 失败" }
    }
}

Write-Host "[5/6] 写入配置、启动脚本和操作手册"
$templateDir = Join-Path $scriptDir "templates"
Copy-Item -LiteralPath (Join-Path $templateDir "config.json") -Destination (Join-Path $dest "config.json") -Force
Copy-Utf8 -Source (Join-Path $templateDir "start_user.ps1") -Dest (Join-Path $dest "start_user.ps1")
Copy-Item -LiteralPath (Join-Path $templateDir "start_web.bat") -Destination (Join-Path $dest "start_web.bat") -Force
Copy-Item -LiteralPath (Join-Path $templateDir "start_web_cmd.bat") -Destination (Join-Path $dest "start_web_cmd.bat") -Force
Copy-Utf8 -Source (Join-Path $scriptDir "templates\安装训练环境.ps1") -Dest (Join-Path $dest "install_train_env.ps1")
Copy-Item -LiteralPath (Join-Path $templateDir "install_train_env.bat") -Destination (Join-Path $dest "install_train_env.bat") -Force
Copy-Utf8 -Source (Join-Path $scriptDir "使用说明.md") -Dest (Join-Path $dest "使用说明.md")

$manual = Join-Path $repoRoot "docs\TRTForge_Web操作手册.md"
$manualImages = Join-Path $repoRoot "docs\images\trtforge_web_manual"
if (Test-Path -LiteralPath $manual) {
    New-Item -ItemType Directory -Path (Join-Path $dest "docs") -Force | Out-Null
    Copy-Utf8 -Source $manual -Dest (Join-Path $dest "docs\TRTForge_Web操作手册.md")
}
$noPsDoc = Join-Path $repoRoot "docs\无PowerShell启动说明.md"
if (Test-Path -LiteralPath $noPsDoc) {
    New-Item -ItemType Directory -Path (Join-Path $dest "docs") -Force | Out-Null
    Copy-Utf8 -Source $noPsDoc -Dest (Join-Path $dest "docs\无PowerShell启动说明.md")
}
if (Test-Path -LiteralPath $manualImages) {
    & robocopy $manualImages (Join-Path $dest "docs\images\trtforge_web_manual") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
}

$stamp = @"
打包时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
编译解释器: $python
CUDA: 11.8
说明: 本目录是给使用人员的安装包，不是源码仓库。
"@
[System.IO.File]::WriteAllText((Join-Path $dest "BUILD.txt"), $stamp, $utf8)

Write-Host "[6/6] 检查有没有把源码打进去"
$leaks = Get-ChildItem -LiteralPath $dest -Recurse -File | Where-Object {
    $_.Name -in @(
        "app.py", "config.py", "infer.py", "frozen_entry.py",
        "tf_yolo_launcher_bin.py", "tf_anomalib_launcher_bin.py"
    )
}
if ($leaks) {
    $leaks | ForEach-Object { Write-Host $_.FullName }
    throw "安装包里出现了源码文件，已停止。请不要把这个目录发出去。"
}
foreach ($stubName in @("tf_yolo_launcher.py", "tf_anomalib_launcher.py")) {
    $stubPath = Join-Path $dest "launchers\$stubName"
    $lines = @(Get-Content -LiteralPath $stubPath)
    if ($lines.Count -gt 12) { throw "$stubName 不是短启动桩，拒绝打包。" }
}

if (-not $SkipZip) {
    $zip = Join-Path $OutDir "TRTForgeWeb_cuda118.zip"
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    tar -a -c -f $zip -C $OutDir "TRTForgeWeb"
    if ($LASTEXITCODE -ne 0) { throw "压缩失败" }
    Write-Host "压缩包: $zip"
}

Write-Host ""
Write-Host "安装目录: $dest" -ForegroundColor Green
Write-Host "把整个 TRTForgeWeb 文件夹交给对方。对方先运行 install_train_env.bat，再双击 start_web.bat。"
