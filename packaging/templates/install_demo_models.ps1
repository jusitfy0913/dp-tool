# -*- coding: utf-8 -*-
# 把示例模型按软件的扫描结构装进安装目录。
# 由 install_demo_models.bat 调用；也可以手工跑：
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\install_demo_models.ps1 -InstallDir D:\TRTForgeWeb
#
# 只做三件事：找软件目录 → 按 manifest.json 落位并校验 → 提示下一步编引擎。
# 不碰软件原有的任何文件（不覆盖 exe、不删已有引擎），所以重复执行是安全的。

param(
    [string]$InstallDir = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PayloadRoot = Join-Path $Root "payload"
$ManifestPath = Join-Path $Root "manifest.json"

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    Write-Host "找不到 manifest.json，这个文件夹可能没解压完整：$Root" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path -LiteralPath $PayloadRoot)) {
    Write-Host "找不到 payload 文件夹，这个文件夹可能没解压完整：$PayloadRoot" -ForegroundColor Red
    exit 1
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json

function Test-InstallDir {
    <#
      判定依据只看一个文件：app\TRTForgeWeb.exe。
      比检查 config.json 或 start_web.bat 更可靠——那两个文件用户可能改名或删掉，
      而 exe 是软件本体，缺了它装哪都不对。
    #>
    param([string]$Path)
    if (-not $Path) { return $false }
    return (Test-Path -LiteralPath (Join-Path $Path "app\TRTForgeWeb.exe"))
}

function Find-InstallDir {
    <#
      先猜常见的三种摆法（旁边同名目录 / 上一层 / 上两层的 TRTForgeWeb），
      再扫一遍同级目录里所有含 app\TRTForgeWeb.exe 的文件夹。
      都找不到才问用户，而不是一上来就让人手打路径。
    #>
    $here = $Root
    $parent = Split-Path -Parent $here
    $grand = Split-Path -Parent $parent
    $candidates = @(
        (Join-Path $parent "TRTForgeWeb"),
        $parent,
        (Join-Path $grand "TRTForgeWeb"),
        $grand
    )
    foreach ($candidate in $candidates) {
        if (Test-InstallDir -Path $candidate) { return $candidate }
    }
    try {
        $siblings = Get-ChildItem -LiteralPath $parent -Directory -ErrorAction Stop
    } catch {
        $siblings = @()
    }
    foreach ($dir in $siblings) {
        if (Test-InstallDir -Path $dir.FullName) { return $dir.FullName }
    }
    return $null
}

if (-not (Test-InstallDir -Path $InstallDir)) {
    if ($InstallDir) {
        Write-Host "这个路径不像软件目录（里面没有 app\TRTForgeWeb.exe）：$InstallDir" -ForegroundColor Yellow
    }
    $InstallDir = Find-InstallDir
}

while (-not (Test-InstallDir -Path $InstallDir)) {
    Write-Host ""
    Write-Host "没有自动找到软件目录。要的是**软件文件夹**，里面应该有 app\TRTForgeWeb.exe。" -ForegroundColor Yellow
    Write-Host "例如：D:\TRTForgeWeb" -ForegroundColor Yellow
    $typed = Read-Host "请粘贴软件文件夹的完整路径（直接回车退出）"
    $typed = "$typed".Trim().Trim('"').Trim("'")
    if (-not $typed) {
        Write-Host "已取消，什么都没改。" -ForegroundColor Yellow
        exit 1
    }
    $InstallDir = $typed
}

$InstallDir = (Resolve-Path -LiteralPath $InstallDir).Path
Write-Host ""
Write-Host "软件目录：$InstallDir"
Write-Host "示例模型：$Root"
Write-Host ""

Write-Host "开始复制并校验……"
$copied = 0
$failed = @()
$modelSummary = @()

foreach ($model in @($manifest.models)) {
    $okFiles = 0
    $onnxList = @()
    foreach ($file in @($model.files)) {
        $relative = [string]$file.path
        $source = Join-Path $PayloadRoot $relative
        $target = Join-Path $InstallDir $relative
        $targetDir = Split-Path -Parent $target

        if (-not (Test-Path -LiteralPath $source)) {
            $failed += "$relative（包里缺这个文件）"
            continue
        }
        if (-not (Test-Path -LiteralPath $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        try {
            Copy-Item -LiteralPath $source -Destination $target -Force
        } catch {
            $failed += "$relative（复制失败：$($_.Exception.Message)）"
            continue
        }

        # 校验：拷过去的字节数 + SHA256 都要对。机械硬盘/网盘同步出错时，
        # 只比大小会漏掉"大小对内容坏"的情况，而坏 ONNX 编出来的引擎没人敢用。
        $targetInfo = Get-Item -LiteralPath $target
        if ([int64]$targetInfo.Length -ne [int64]$file.bytes) {
            $failed += "$relative（大小不符：期望 $($file.bytes)，实际 $($targetInfo.Length)）"
            continue
        }
        $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($hash -ne ([string]$file.sha256).ToUpper()) {
            $failed += "$relative（校验和不符，文件可能损坏）"
            continue
        }

        if ($relative.ToLower().EndsWith(".onnx")) { $onnxList += $target }
        $okFiles += 1
        $copied += 1
    }

    if ($onnxList.Count -gt 0) {
        # anomalib 的降级推理会回查 metadata.json 里的 onnx_path（原始训练机路径已经失效），
        # 这里改成这台机器上的真实绝对路径，换机器也能用。
        $metaFile = ""
        foreach ($file in @($model.files)) {
            # 这里必须先把路径取成字符串再比：写成 `[string]$file.path.ToLower().EndsWith(...)`
            # 会把 EndsWith 的布尔结果转成 "True"/"False"，而非空字符串恒为真——
            # 于是 args.yaml 也会被当成 metadata 去 ConvertFrom-Json（实测报
            # 「Invalid JSON primitive: task.」）。若那个文件恰好是合法 JSON，还会被覆盖写坏。
            $relativePath = [string]$file.path
            if ($relativePath.ToLower().EndsWith("metadata.json")) {
                $metaFile = Join-Path $InstallDir $relativePath
            }
        }
        # 先判空再 Test-Path：`Test-Path -LiteralPath ""` 在 $ErrorActionPreference='Stop'
        # 下会直接终止脚本（YOLO 模型没有 metadata.json，$metaFile 就是空串）。
        if ($metaFile -and (Test-Path -LiteralPath $metaFile)) {
            try {
                $meta = Get-Content -LiteralPath $metaFile -Raw -Encoding UTF8 | ConvertFrom-Json
                $meta | Add-Member -NotePropertyName "onnx_path" -NotePropertyValue $onnxList[0] -Force
                $metaJson = $meta | ConvertTo-Json -Depth 8
                [System.IO.File]::WriteAllText($metaFile, $metaJson, (New-Object System.Text.UTF8Encoding $false))
            } catch {
                Write-Host "  [WARN] metadata.json 里的 onnx_path 没写进去：$($_.Exception.Message)" -ForegroundColor Yellow
            }
        }
    }

    $modelSummary += [pscustomobject]@{
        title = [string]$model.title
        kind = [string]$model.kind
        files = $okFiles
        total = @($model.files).Count
        onnx = ($onnxList -join "; ")
    }
}

Write-Host ""
foreach ($row in $modelSummary) {
    $mark = "OK"
    if ($row.files -lt $row.total) { $mark = "部分失败" }
    Write-Host ("  [{0}] {1}（{2}/{3} 个文件）" -f $mark, $row.title, $row.files, $row.total)
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "以下文件没装好：" -ForegroundColor Red
    foreach ($item in $failed) { Write-Host "  - $item" -ForegroundColor Red }
}

$record = [pscustomobject]@{
    installed_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    install_dir = $InstallDir
    app_version = [string]$manifest.app_version
    file_count = $copied
    failed = $failed
    models = $modelSummary
}
$launchersDir = Join-Path $InstallDir "launchers"
if (-not (Test-Path -LiteralPath $launchersDir)) { New-Item -ItemType Directory -Path $launchersDir -Force | Out-Null }
[System.IO.File]::WriteAllText(
    (Join-Path $launchersDir "demo_models_installed.json"),
    ($record | ConvertTo-Json -Depth 6),
    (New-Object System.Text.UTF8Encoding $false)
)

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "复制完成，全部校验通过。" -ForegroundColor Green
} else {
    Write-Host "复制完成，但有文件没通过校验，请把上面的红字发给提供安装包的人。" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "下一步：在对方显卡上把 ONNX 编成引擎（引擎不能跨机器拷贝）"
Write-Host ""
Write-Host "  1. 双击软件目录里的 start_web.bat"
Write-Host "  2. 打开左侧「引擎构建」页"
Write-Host "  3. 「ONNX 文件」填下面任意一行，「输出引擎」留空，精度选 FP16，点开始构建"
Write-Host ""
foreach ($row in $modelSummary) {
    if ($row.onnx) {
        Write-Host ("    {0}" -f $row.onnx) -ForegroundColor Cyan
    }
}
Write-Host ""
Write-Host "  4. 构建完成后回到「推理演示」页，模型下拉里就会出现这些示例模型"
Write-Host ""
Write-Host "提示：已经编过引擎的模型会被跳过，重复执行本脚本不会破坏已有的引擎。"
