# Zip the three inference DLL sources (3-cpp-tensorrt + ocr) plus packaging scripts and DLL usage docs.
# Run from anywhere (script discovers repo via docs/TRTForge_yolo_infer*.md):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\4-qt-studio\inference\package_gui_inference_full.ps1
# Optional:
#   .\package_gui_inference_full.ps1 -OutDir D:\dist -ZipName my_pack

param(
  [string]$OutDir = "",
  [string]$ZipName = ""
)

$ErrorActionPreference = "Stop"

$inferenceRoot = [System.IO.Path]::GetDirectoryName($MyInvocation.MyCommand.Path)

function Find-RepoRootWithSdkDoc {
  param([string]$StartDir)
  $d = $StartDir
  for ($i = 0; $i -lt 16; $i++) {
    $docsDir = Join-Path $d "docs"
    if (Test-Path -LiteralPath $docsDir) {
      $hits = @(Get-ChildItem -LiteralPath $docsDir -Filter "TRTForge_yolo_infer*.md" -File -ErrorAction SilentlyContinue)
      if ($hits.Count -ge 1) {
        return @{ Root = $d; SdkDoc = $hits[0].FullName }
      }
    }
    $parentInfo = [System.IO.Directory]::GetParent($d)
    if (-not $parentInfo) { break }
    $parent = $parentInfo.FullName
    if (-not $parent -or $parent -eq $d) { break }
    $d = $parent
  }
  return $null
}

$found = Find-RepoRootWithSdkDoc -StartDir $inferenceRoot
if (-not $found) {
  throw "Could not find docs/TRTForge_yolo_infer*.md walking up from: $inferenceRoot"
}
$repoRoot = $found.Root
$sdkDoc = $found.SdkDoc
$repoDocs = Join-Path $repoRoot "docs"

if (-not (Test-Path -LiteralPath $inferenceRoot)) {
  throw "Missing inference root: $inferenceRoot"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = $repoRoot
}

if ([string]::IsNullOrWhiteSpace($ZipName)) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $ZipName = "trtforge_gui_inference_full_$ts"
}

$staging = Join-Path $env:TEMP ("trtforge_gui_inference_pack_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $staging | Out-Null

try {
  $destRoot = Join-Path $staging $ZipName
  New-Item -ItemType Directory -Path $destRoot | Out-Null

  $destInference = Join-Path $destRoot "inference"
  New-Item -ItemType Directory -Path $destInference | Out-Null
  Copy-Item -LiteralPath (Join-Path $repoRoot "3-cpp-tensorrt\yolo_infer") -Destination (Join-Path $destInference "yolo_infer") -Recurse -Force
  Copy-Item -LiteralPath (Join-Path $repoRoot "3-cpp-tensorrt\anomalib_infer") -Destination (Join-Path $destInference "anomalib_infer") -Recurse -Force
  Copy-Item -LiteralPath (Join-Path $repoRoot "ocr\ocr_infer") -Destination (Join-Path $destInference "ocr_infer") -Recurse -Force
  Copy-Item -LiteralPath $inferenceRoot -Destination (Join-Path $destInference "pack-scripts") -Recurse -Force

  $destDocs = Join-Path $destRoot "docs"
  New-Item -ItemType Directory -Path $destDocs | Out-Null
  Copy-Item -LiteralPath $sdkDoc -Destination (Join-Path $destDocs "TRTForge_inference_dll_SDK_zh-CN.md") -Force

  $yoloDocs = @(Get-ChildItem -LiteralPath $repoDocs -Filter "trtexec_gui_YOLO*.md" -File -ErrorAction SilentlyContinue)
  if ($yoloDocs.Count -ge 1) {
    Copy-Item -LiteralPath $yoloDocs[0].FullName -Destination (Join-Path $destDocs "TRTForge_YOLO_TRT_inference_detail_zh-CN.md") -Force
  }

  $readme = @"
trtforge inference DLL full source pack + docs (UTF-8)

inference/
  yolo_infer/       YOLO TensorRT DLL source (yi_* C API), see include/yolo_infer_c.h
  anomalib_infer/   Anomalib TRT DLL source (ai_* C API), see README.md + include/anomalib_infer_c.h
  ocr_infer/        PaddleOCR C++ DLL source (oi_* C API), see README.md + include/ocr_infer_c.h
  package_*.ps1     Pack scripts

docs/
  TRTForge_inference_dll_SDK_zh-CN.md          yolo_infer + anomalib_infer + ocr_infer call order, memory, deps, pack
  TRTForge_YOLO_TRT_inference_detail_zh-CN.md  (if present) YOLO pipeline deep doc

Build: add 3-cpp-tensorrt/yolo_infer, 3-cpp-tensorrt/anomalib_infer and ocr/ocr_infer to the 4-qt-studio CMake project (TRTFORGE_INFER_BLOCK).
Runtime: TensorRT + CUDA + OpenCV + Paddle Inference DLLs on PATH or next to TRTForge.exe / your host EXE.
"@
  Set-Content -LiteralPath (Join-Path $destRoot "README_PACKAGE.txt") -Value $readme -Encoding UTF8

  $zipPath = Join-Path $OutDir "$ZipName.zip"
  if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
  }
  Compress-Archive -Path $destRoot -DestinationPath $zipPath -CompressionLevel Optimal -Force
  Write-Host "OK: $zipPath"
}
finally {
  if (Test-Path -LiteralPath $staging) {
    Remove-Item -LiteralPath $staging -Recurse -Force
  }
}
