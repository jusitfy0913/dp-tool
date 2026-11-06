# -*- coding: utf-8 -*-
# 下载 ONNX Runtime 的 Windows x64 预编译包，供 onnx_infer 子工程编译与运行。
#
# 为什么用 NuGet 包而不是 GitHub Release：
#   1. NuGet 的 nupkg 就是 zip，解开后 include 与原生 DLL 在同一份产物里，版本天然对齐；
#   2. GitHub Release 的 GPU 包下载地址经常 302 到对象存储，国内网络容易断；
#   3. 同一版本号在 NuGet 上是不可变的，比 release 附件更适合写进脚本。
#
# 版本选择：1.16.3 是最后一个默认按 CUDA 11.8 + cuDNN 8 构建的稳定版，
# 与本项目的 TensorRT / CUDA 工具链一致。1.17 之后默认 CUDA 12，会与 nvinfer 抢 CUDA 运行时。
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_onnxruntime.ps1
#   powershell ... -File scripts\fetch_onnxruntime.ps1 -Variant cpu      # 只要 CPU，包体约 10 MB
#   powershell ... -File scripts\fetch_onnxruntime.ps1 -Force            # 重新下载
#
# 产物：
#   3-cpp-tensorrt\onnx_infer\third_party\onnxruntime\
#     include\*.h          编译期头文件
#     lib\*.dll            运行期动态库（onnxruntime.dll 与 provider）
#     VERSION.txt          版本与来源，便于排错时确认到底加载了哪一份

param(
    [ValidateSet("gpu", "cpu")]
    [string]$Variant = "gpu",
    [string]$Version = "1.16.3",
    [string]$Dest = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Dest) {
    $Dest = Join-Path $RepoRoot "3-cpp-tensorrt\onnx_infer\third_party\onnxruntime"
}

# NuGet 包名与 nupkg 内的原生目录固定，不随版本变化
$PackageName = if ($Variant -eq "gpu") { "Microsoft.ML.OnnxRuntime.Gpu" } else { "Microsoft.ML.OnnxRuntime" }
$NuGetUrl = "https://www.nuget.org/api/v2/package/$PackageName/$Version"
# 备用源：GitHub Release 的 Windows 产物（CPU 与 GPU 的资产名不同）
$GitHubUrl = if ($Variant -eq "gpu") {
    "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-gpu-$Version.zip"
} else {
    "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-$Version.zip"
}

Write-Host "ONNX Runtime 准备"
Write-Host "  variant : $Variant"
Write-Host "  version : $Version"
Write-Host "  dest    : $Dest"

$versionFile = Join-Path $Dest "VERSION.txt"
if ((Test-Path $versionFile) -and -not $Force) {
    $existing = Get-Content $versionFile -Raw -Encoding UTF8
    if ($existing -match [regex]::Escape("$PackageName $Version")) {
        Write-Host "已存在同一版本，跳过下载。需要重下请加 -Force。" -ForegroundColor Green
        exit 0
    }
}

# ---------------------------------------------------------------------------
# 下载（先 NuGet，再 GitHub），任何一步成功即可
# ---------------------------------------------------------------------------
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ort_fetch_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$archive = Join-Path $tempRoot "ort.zip"

function Get-Remote([string]$Url, [string]$OutFile) {
    # curl.exe 在 Win10 1803+ 自带，比 Invoke-WebRequest 更稳（后者会走 IE 引擎并受代理影响）
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -s -L --fail --max-time 1800 -o $OutFile $Url
        return ($LASTEXITCODE -eq 0) -and (Test-Path $OutFile) -and ((Get-Item $OutFile).Length -gt 0)
    }
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -TimeoutSec 1800
        return (Test-Path $OutFile) -and ((Get-Item $OutFile).Length -gt 0)
    } catch {
        return $false
    }
}

$downloaded = $false
foreach ($source in @(@{ Name = "nuget"; Url = $NuGetUrl }, @{ Name = "github"; Url = $GitHubUrl })) {
    Write-Host "下载 $($source.Name)：$($source.Url)"
    if (Get-Remote -Url $source.Url -OutFile $archive) {
        Write-Host "  完成，$([math]::Round((Get-Item $archive).Length / 1MB, 1)) MB" -ForegroundColor Green
        $downloaded = $true
        $usedSource = $source.Name
        break
    }
    Write-Warning "  $($source.Name) 下载失败，尝试下一个源"
}

if (-not $downloaded) {
    Remove-Item $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    throw "ONNX Runtime $Version ($Variant) 下载失败。请检查网络，或手动下载后解压到 $Dest。"
}

# ---------------------------------------------------------------------------
# 解压并整理成 include / lib 两个目录
# ---------------------------------------------------------------------------
$extract = Join-Path $tempRoot "extract"
Expand-Archive -Path $archive -DestinationPath $extract -Force

$includeSrc = Get-ChildItem $extract -Recurse -Directory -Filter "include" |
    Where-Object { Test-Path (Join-Path $_.FullName "onnxruntime_c_api.h") } |
    Select-Object -First 1
if (-not $includeSrc) {
    Remove-Item $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    throw "压缩包里找不到 onnxruntime_c_api.h，包结构可能与预期不同。"
}

# NuGet 包里同时有 win-arm64 / win-x64 / win-x86 三份原生目录，
# 必须按 win-x64 精确匹配，否则会把 ARM64 的 DLL 复制到 x64 工程里
# （表现为 Python 报 WinError 193 / C++ 报找不到 OrtGetApiBase）。
$nativeSrc = Get-ChildItem $extract -Recurse -Directory -Filter "native" |
    Where-Object { $_.FullName -match "win-x64" -and (Test-Path (Join-Path $_.FullName "onnxruntime.dll")) } |
    Select-Object -First 1
if (-not $nativeSrc) {
    # 目录结构不同的包版本，退化为按 PE 架构筛选
    $nativeSrc = Get-ChildItem $extract -Recurse -Directory -Filter "native" |
        Where-Object { Test-Path (Join-Path $_.FullName "onnxruntime.dll") } |
        Where-Object {
            $candidate = Join-Path $_.FullName "onnxruntime.dll"
            $stream = [System.IO.File]::OpenRead($candidate)
            $buffer = New-Object byte[] 6
            $stream.Position = 0x100
            $stream.Read($buffer, 0, 6) | Out-Null
            $stream.Close()
            # 0x8664 即 IMAGE_FILE_MACHINE_AMD64
            ([BitConverter]::ToUInt16($buffer, 4) -eq 0x8664)
        } |
        Select-Object -First 1
}
if (-not $nativeSrc) {
    Remove-Item $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    throw "压缩包里找不到 win-x64 的 onnxruntime.dll，包结构可能与预期不同。"
}

New-Item -ItemType Directory -Path (Join-Path $Dest "include") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Dest "lib") -Force | Out-Null

Copy-Item (Join-Path $includeSrc.FullName "*.h") (Join-Path $Dest "include") -Force
Copy-Item (Join-Path $nativeSrc.FullName "*.dll") (Join-Path $Dest "lib") -Force

$dlls = Get-ChildItem (Join-Path $Dest "lib") -Filter "*.dll" | Sort-Object Name
$headers = Get-ChildItem (Join-Path $Dest "include") -Filter "*.h" | Sort-Object Name

@(
    "package=$PackageName"
    "version=$Version"
    "variant=$Variant"
    "source=$usedSource"
    "fetched=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "dlls=$($dlls.Name -join ',')"
) | Set-Content -Path $versionFile -Encoding UTF8

Remove-Item $tempRoot -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "完成：$Dest" -ForegroundColor Green
Write-Host "  头文件 $($headers.Count) 个"
$dlls | ForEach-Object { Write-Host ("  {0,-42} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
if ($Variant -eq "gpu") {
    Write-Host ""
    Write-Host "提示：GPU 包要求 CUDA 11.8 与 cuDNN 8 在 PATH 或 bin 目录中。" -ForegroundColor Yellow
}
