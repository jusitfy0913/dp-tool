# Package TRTForge development kit.
# Inference: prebuilt DLL + import lib + public headers only (no infer source).
# Training: 1-python-yolo and 2-python-anomaly stay beside 4-qt-studio.
#
# Usage (from repo root or any cwd):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\3-cpp-tensorrt\trtexec\package_devkit.ps1
# Optional:
#   .\package_devkit.ps1 -OutDir D:\dist -ZipName trtforge_devkit_20260902 -SkipZip

param(
    [string]$OutDir = "",
    [string]$ZipName = "",
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Invoke-RobocopySafe {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Dest,
        [string[]]$ExcludeDirs = @(),
        [string[]]$ExcludeFiles = @()
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Missing source: $Source"
    }
    New-Item -ItemType Directory -Path $Dest -Force | Out-Null
    $args = @($Source, $Dest, "/E", "/XJ", "/R:1", "/W:1", "/MT:8", "/NFL", "/NDL", "/NP")
    if ($ExcludeDirs.Count -gt 0) {
        $args += "/XD"
        $args += $ExcludeDirs
    }
    if ($ExcludeFiles.Count -gt 0) {
        $args += "/XF"
        $args += $ExcludeFiles
    }
    & robocopy @args | Out-Host
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed with exit code $code from $Source to $Dest"
    }
    return $code
}

function Remove-IfExists {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Write-Utf8File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Content, $utf8)
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trtexecRoot = $scriptDir

# 仓库根目录：从脚本位置逐级向上查找标记目录，避免依赖固定层级
function Find-RepoRoot {
    param([string]$StartDir, [int]$MaxDepth = 8)
    $dir = $StartDir
    for ($i = 0; $i -lt $MaxDepth; $i++) {
        if ((Test-Path -LiteralPath (Join-Path $dir "4-qt-studio")) -and
            (Test-Path -LiteralPath (Join-Path $dir "1-python-yolo"))) {
            return $dir
        }
        $parent = Split-Path -Parent $dir
        if (-not $parent -or $parent -eq $dir) { break }
        $dir = $parent
    }
    return $null
}

$repoRoot = Find-RepoRoot -StartDir $scriptDir
if (-not $repoRoot) {
    throw "Cannot locate repo root (need 4-qt-studio and 1-python-yolo siblings) walking up from: $scriptDir"
}
$guiRoot = Join-Path $repoRoot "4-qt-studio"
$binRoot = Join-Path $guiRoot "bin"

$requiredDlls = @(
    (Join-Path $binRoot "yolo_infer.dll"),
    (Join-Path $binRoot "anomalib_infer.dll"),
    (Join-Path $binRoot "ocr_infer.dll")
)
foreach ($dll in $requiredDlls) {
    if (-not (Test-Path -LiteralPath $dll)) {
        throw "Missing prebuilt inference DLL: $dll"
    }
}

$yoloLib = Join-Path $binRoot "yolo_infer.lib"
if (-not (Test-Path -LiteralPath $yoloLib)) {
    $yoloLib = Join-Path $binRoot "Release\yolo_infer.lib"
}
$anomLib = Join-Path $binRoot "anomalib_infer.lib"
if (-not (Test-Path -LiteralPath $anomLib)) {
    $anomLib = Join-Path $binRoot "Release\anomalib_infer.lib"
}
$ocrLib = Join-Path $binRoot "ocr_infer.lib"
if (-not (Test-Path -LiteralPath $ocrLib)) {
    $ocrLib = Join-Path $binRoot "Release\ocr_infer.lib"
}
foreach ($lib in @($yoloLib, $anomLib, $ocrLib)) {
    if (-not (Test-Path -LiteralPath $lib)) {
        throw "Missing inference import lib: $lib"
    }
}

$ultraRoot = Join-Path $repoRoot "1-python-yolo\ultralytics-main"
$anomalibMain = Join-Path $repoRoot "2-python-anomaly\anomalib-main"
$yoloLauncher = Join-Path $repoRoot "1-python-yolo\launcher"
$anomalyLauncher = Join-Path $repoRoot "2-python-anomaly\launcher"
if (-not (Test-Path -LiteralPath $ultraRoot)) {
    throw "Missing training project: $ultraRoot"
}
if (-not (Test-Path -LiteralPath $anomalibMain)) {
    throw "Missing training project: $anomalibMain"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "dist"
}
if ([string]::IsNullOrWhiteSpace($ZipName)) {
    $ZipName = "trtforge_devkit_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$destRoot = Join-Path $OutDir $ZipName
if (Test-Path -LiteralPath $destRoot) {
    Remove-Item -LiteralPath $destRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $destRoot | Out-Null

Write-Host "[devkit] copy 4-qt-studio -> $destRoot\4-qt-studio"
$destGui = Join-Path $destRoot "4-qt-studio"
Invoke-RobocopySafe -Source $guiRoot -Dest $destGui -ExcludeDirs @(
    "build", "logs", "dist", ".vs", ".git", "runs"
) -ExcludeFiles @(
    "*.pdb", "*.ilk", "*.exp", "*.old", "paddle_inference (1).zip"
)

$noticeSrc = Join-Path $guiRoot "inference\README_DEVKIT_INFER.md"
$headerModules = @(
    @{ Src = Join-Path $repoRoot "3-cpp-tensorrt\yolo_infer"; Dest = Join-Path $destRoot "3-cpp-tensorrt\yolo_infer" },
    @{ Src = Join-Path $repoRoot "3-cpp-tensorrt\anomalib_infer"; Dest = Join-Path $destRoot "3-cpp-tensorrt\anomalib_infer" },
    @{ Src = Join-Path $repoRoot "ocr\ocr_infer"; Dest = Join-Path $destRoot "ocr\ocr_infer" }
)
foreach ($mod in $headerModules) {
    $srcInc = Join-Path $mod.Src "include"
    if (-not (Test-Path -LiteralPath $srcInc)) {
        throw "Missing inference headers: $srcInc"
    }
    Invoke-RobocopySafe -Source $srcInc -Dest (Join-Path $mod.Dest "include")
    if (Test-Path -LiteralPath $noticeSrc) {
        Copy-Item -LiteralPath $noticeSrc -Destination (Join-Path $mod.Dest "README_DEVKIT.md") -Force
    }
}

$destBin = Join-Path $destGui "bin"
New-Item -ItemType Directory -Path $destBin -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $binRoot "yolo_infer.dll") -Destination (Join-Path $destBin "yolo_infer.dll") -Force
Copy-Item -LiteralPath (Join-Path $binRoot "anomalib_infer.dll") -Destination (Join-Path $destBin "anomalib_infer.dll") -Force
Copy-Item -LiteralPath (Join-Path $binRoot "ocr_infer.dll") -Destination (Join-Path $destBin "ocr_infer.dll") -Force
Copy-Item -LiteralPath $yoloLib -Destination (Join-Path $destBin "yolo_infer.lib") -Force
Copy-Item -LiteralPath $anomLib -Destination (Join-Path $destBin "anomalib_infer.lib") -Force
Copy-Item -LiteralPath $ocrLib -Destination (Join-Path $destBin "ocr_infer.lib") -Force

$cmakePath = Join-Path $destGui "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $cmakePath)) {
    throw "Packaged CMakeLists.txt missing: $cmakePath"
}
$cmakeText = [System.IO.File]::ReadAllText($cmakePath)
$cmakeNorm = $cmakeText.Replace("`r`n", "`n")
$inferCMakePattern = '(?s)# TRTFORGE_INFER_BLOCK_BEGIN.*?# TRTFORGE_INFER_BLOCK_END'
if ($cmakeNorm -notmatch $inferCMakePattern) {
    throw "CMakeLists.txt infer block markers not found; refuse to ship an unpatched GUI project."
}
$newBlock = @"
# TRTFORGE_INFER_BLOCK_BEGIN
# Development kit: inference implementations are prebuilt DLLs only.
function(trtforge_import_prebuilt_infer _name)
    set(_dll "`${CMAKE_CURRENT_SOURCE_DIR}/bin/`${_name}.dll")
    set(_lib "`${CMAKE_CURRENT_SOURCE_DIR}/bin/`${_name}.lib")
    if(NOT EXISTS "`${_lib}")
        set(_lib "`${CMAKE_CURRENT_SOURCE_DIR}/bin/Release/`${_name}.lib")
    endif()
    if(NOT EXISTS "`${_lib}" OR NOT EXISTS "`${_dll}")
        message(FATAL_ERROR
            "[TRTForge] prebuilt `${_name} not found.\n"
            "  expected dll: `${_dll}\n"
            "  expected lib: `${_lib}")
    endif()
    add_library(`${_name} SHARED IMPORTED GLOBAL)
    set_target_properties(`${_name} PROPERTIES
        IMPORTED_IMPLIB "`${_lib}"
        IMPORTED_LOCATION "`${_dll}"
    )
    message(STATUS "[TRTForge] Using prebuilt `${_name}: `${_dll}")
endfunction()

trtforge_import_prebuilt_infer(yolo_infer)
trtforge_import_prebuilt_infer(anomalib_infer)
trtforge_import_prebuilt_infer(ocr_infer)
# TRTFORGE_INFER_BLOCK_END
"@
$cmakeNorm = [regex]::Replace(
    $cmakeNorm,
    $inferCMakePattern,
    $newBlock.TrimEnd()
)
$cmakeNorm = [regex]::Replace(
    $cmakeNorm,
    '(?s)\r?\noption\(TRTFORGE_BUILD_OCR_INFER\r?\n    "Build ocr_infer inside TRTForge solution \(pulls Abseil third-party targets into VS\)"\r?\n    OFF\)\r?\n',
    "`n"
)
$cmakeOut = $cmakeNorm.Replace("`n", "`r`n")
Write-Utf8File -Path $cmakePath -Content $cmakeOut

if (-not (Test-Path -LiteralPath $noticeSrc)) {
    throw "Missing inference notice: $noticeSrc"
}

Write-Host "[devkit] copy 1-python-yolo"
Invoke-RobocopySafe -Source $ultraRoot -Dest (Join-Path $destRoot "1-python-yolo\ultralytics-main") -ExcludeDirs @(
    ".git", "__pycache__", ".pytest_cache", "runs", "datasets"
)
Invoke-RobocopySafe -Source $yoloLauncher -Dest (Join-Path $destRoot "1-python-yolo\launcher")

Write-Host "[devkit] copy 2-python-anomaly (no datasets/results)"
$destAnomalibMain = Join-Path $destRoot "2-python-anomaly\anomalib-main"
Invoke-RobocopySafe -Source $anomalibMain -Dest $destAnomalibMain -ExcludeDirs @(
    ".git", "__pycache__", ".pytest_cache", "datasets", "results", "runs"
)
Invoke-RobocopySafe -Source $anomalyLauncher -Dest (Join-Path $destRoot "2-python-anomaly\launcher")

$destDocs = Join-Path $destRoot "docs"
New-Item -ItemType Directory -Path $destDocs -Force | Out-Null
$srcDocs = Join-Path $repoRoot "docs"
if (Test-Path -LiteralPath $srcDocs) {
    Get-ChildItem -LiteralPath $srcDocs -File -Filter "TRTForge_*.md" | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $destDocs $_.Name) -Force
    }
}
$rootReadme = Join-Path $repoRoot "readme.md"
if (Test-Path -LiteralPath $rootReadme) {
    Copy-Item -LiteralPath $rootReadme -Destination (Join-Path $destRoot "readme.md") -Force
}

$readmeTemplate = Join-Path $trtexecRoot "README_DEVKIT.md"
if (-not (Test-Path -LiteralPath $readmeTemplate)) {
    throw "Missing README_DEVKIT.md next to package_devkit.ps1"
}
$readmeBody = [System.IO.File]::ReadAllText($readmeTemplate)
$readmeStamp = "# Packaged at: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "`r`n`r`n"
Write-Utf8File -Path (Join-Path $destRoot "README_DEVKIT.md") -Content ($readmeStamp + $readmeBody)
Copy-Item -LiteralPath $MyInvocation.MyCommand.Path -Destination (Join-Path $destRoot "package_devkit.ps1") -Force

$missing = @()
foreach ($dllName in @("yolo_infer.dll", "anomalib_infer.dll", "ocr_infer.dll", "yolo_infer.lib", "anomalib_infer.lib", "ocr_infer.lib")) {
    if (-not (Test-Path -LiteralPath (Join-Path $destBin $dllName))) {
        $missing += $dllName
    }
}
foreach ($header in @(
        "3-cpp-tensorrt\yolo_infer\include\yolo_infer_c.h",
        "3-cpp-tensorrt\anomalib_infer\include\anomalib_infer_c.h",
        "ocr\ocr_infer\include\ocr_infer_c.h",
        "1-python-yolo\launcher\tf_yolo_launcher.py",
        "2-python-anomaly\launcher\tf_anomalib_launcher.py"
    )) {
    if (-not (Test-Path -LiteralPath (Join-Path $destRoot $header))) {
        $missing += $header
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $destRoot "1-python-yolo\ultralytics-main"))) {
    $missing += "1-python-yolo/ultralytics-main"
}
if (-not (Test-Path -LiteralPath (Join-Path $destRoot "2-python-anomaly\anomalib-main"))) {
    $missing += "2-python-anomaly/anomalib-main"
}
$leakRoots = @(
    (Join-Path $destRoot "3-cpp-tensorrt"),
    (Join-Path $destRoot "ocr\ocr_infer")
)
$leaked = @()
foreach ($leakRoot in $leakRoots) {
    if (Test-Path -LiteralPath $leakRoot) {
        $leaked += @(Get-ChildItem -LiteralPath $leakRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -match '^\.(cpp|cu|cuh|cc|cxx)$' })
    }
}
if ($leaked.Count -gt 0) {
    throw ("Inference source leaked into package: " + ($leaked.FullName -join ", "))
}
if ($missing.Count -gt 0) {
    throw ("Package validation failed, missing: " + ($missing -join ", "))
}

$zipPath = Join-Path $OutDir "$ZipName.zip"
if (-not $SkipZip) {
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Write-Host "[devkit] creating zip $zipPath"
    Push-Location $OutDir
    try {
        & tar -a -c -f "$ZipName.zip" $ZipName
        if ($LASTEXITCODE -ne 0) {
            throw "tar zip failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "[devkit] done"
Write-Host "[devkit] folder: $destRoot"
if (-not $SkipZip) {
    Write-Host "[devkit] zip:    $zipPath"
}
