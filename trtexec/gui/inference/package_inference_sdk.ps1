# Package yolo_infer + anomalib_infer + ocr_infer sources, headers, CMake, examples, and the SDK doc into one zip.
# Run from repo root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\trtexec\gui\inference\package_inference_sdk.ps1
# Optional:
#   .\trtexec\gui\inference\package_inference_sdk.ps1 -OutDir D:\dist -ZipName my_sdk

param(
  [string]$OutDir = "",
  [string]$ZipName = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = [System.IO.Path]::GetDirectoryName($MyInvocation.MyCommand.Path)
$yoloDir = Join-Path $scriptDir "yolo_infer"
$anomDir = Join-Path $scriptDir "anomalib_infer"
$ocrDir = Join-Path $scriptDir "ocr_infer"

function Find-RepoRootWithSdkDoc {
  param([string]$StartDir)
  $d = $StartDir
  for ($i = 0; $i -lt 12; $i++) {
    $docsDir = Join-Path $d "docs"
    if (Test-Path -LiteralPath $docsDir) {
      $hits = @(Get-ChildItem -LiteralPath $docsDir -Filter "TRTForge_yolo_infer*.md" -File -ErrorAction SilentlyContinue)
      if ($hits.Count -ge 1) {
        return @{ Root = $d; Doc = $hits[0].FullName }
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

$found = Find-RepoRootWithSdkDoc -StartDir $scriptDir
if (-not $found) {
  throw "Could not find docs/TRTForge_yolo_infer*.md by walking up from: $scriptDir"
}
$repoRoot = $found.Root
$docPath = $found.Doc

foreach ($p in @($yoloDir, $anomDir, $ocrDir)) {
  if (-not (Test-Path -LiteralPath $p)) {
    throw "Missing path: $p"
  }
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = $repoRoot
}

if ([string]::IsNullOrWhiteSpace($ZipName)) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $ZipName = "trtforge_inference_dll_sdk_$ts"
}

$staging = Join-Path $env:TEMP ("trtforge_inference_pack_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $staging | Out-Null

try {
  $destRoot = Join-Path $staging $ZipName
  New-Item -ItemType Directory -Path $destRoot | Out-Null

  $destYolo = Join-Path $destRoot "yolo_infer"
  $destAnom = Join-Path $destRoot "anomalib_infer"
  $destOcr = Join-Path $destRoot "ocr_infer"
  Copy-Item -LiteralPath $yoloDir -Destination $destYolo -Recurse -Force
  Copy-Item -LiteralPath $anomDir -Destination $destAnom -Recurse -Force
  Copy-Item -LiteralPath $ocrDir -Destination $destOcr -Recurse -Force
  Copy-Item -LiteralPath $docPath -Destination (Join-Path $destRoot "TRTForge_inference_dll_SDK_zh-CN.md") -Force
  Copy-Item -LiteralPath $MyInvocation.MyCommand.Path -Destination (Join-Path $destRoot "package_inference_sdk.ps1") -Force

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
