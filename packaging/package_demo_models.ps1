# -*- coding: utf-8 -*-
# 打「示例模型包」：和软件安装包分开，装不装模型由用户自己决定。
#
# 为什么要拆开（而不是塞进软件包）：
#   1) 软件包已经 1.8 GB，再塞模型对方下载更久，而很多人只需要软件；
#   2) 示例模型是「可替换的数据」，会随演示场景变，不该让软件包跟着重打；
#   3) 只带 ONNX 时引擎要在**对方显卡**上编一次才准，所以模型包必须能单独更新。
#
# 包里只有 ONNX，没有 .engine：TensorRT 引擎绑定显卡型号与 TRT 版本，
# 在开发机编好的引擎换台机器大概率报序列化头不匹配。让对方用「引擎构建」页
# 当场编一次（PCB 模型实测约 474 秒），这条路才稳。
#
# 在 dp-tool 目录执行：
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\package_demo_models.ps1
#
# 可选：
#   -OutDir D:\dist          输出目录（默认 <仓库>\dist）
#   -Manifest <json>         换一份模型清单（默认 packaging\models_manifest.json）
#   -SkipZip                 只生成文件夹，不压缩（调试布局时用）

param(
    [string]$OutDir = "",
    [string]$ManifestPath = "",
    [switch]$SkipZip
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

function Read-AppVersion {
    param([string]$RepoRoot)
    $configPy = Join-Path $RepoRoot "6-web\backend\config.py"
    if (-not (Test-Path -LiteralPath $configPy)) { return "0.0.0" }
    $match = Select-String -LiteralPath $configPy -Pattern '^APP_VERSION\s*=\s*"([^"]+)"' |
        Select-Object -First 1
    if ($match) { return $match.Matches[0].Groups[1].Value }
    return "0.0.0"
}

function Copy-FileRecord {
    <#
      复制一个文件并记一条清单项。
      清单里的 path 是**相对安装目录**的路径，安装脚本据此落位；
      sha256 用来在用户机器上判断"拷过去的东西有没有坏"。
    #>
    param(
        [string]$Source,
        [string]$DestRoot,
        [string]$Relative,
        [System.Collections.ArrayList]$Sink
    )
    if (-not (Test-Path -LiteralPath $Source)) { throw "找不到源文件：$Source" }
    $destPath = Join-Path $DestRoot $Relative
    $parent = Split-Path -Parent $destPath
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $destPath -Force
    $item = Get-Item -LiteralPath $destPath
    $hash = (Get-FileHash -LiteralPath $destPath -Algorithm SHA256).Hash
    [void]$Sink.Add([pscustomobject]@{
        path = $Relative.Replace("/", "\")
        bytes = [int64]$item.Length
        sha256 = $hash
    })
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# 密钥扫描器：与软件包共用同一份实现（见 packaging\secret_scan.ps1）。
. (Join-Path $scriptDir "secret_scan.ps1")

$repoRoot = Find-RepoRoot -StartDir $scriptDir
if (-not $repoRoot) { throw "找不到仓库根目录（需要 6-web 和 4-qt-studio）。" }

if (-not $ManifestPath) { $ManifestPath = Join-Path $scriptDir "models_manifest.json" }
if (-not (Test-Path -LiteralPath $ManifestPath)) { throw "找不到示例模型清单：$ManifestPath" }
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $repoRoot "dist" }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$spec = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $spec.models -or @($spec.models).Count -eq 0) { throw "清单里没有 models 条目：$ManifestPath" }

$appVersion = Read-AppVersion -RepoRoot $repoRoot
$dest = Join-Path $OutDir "TRTForgeWeb_models_demo"
$payload = Join-Path $dest "payload"
if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
New-Item -ItemType Directory -Path $payload -Force | Out-Null

$entries = New-Object System.Collections.ArrayList
$totalBytes = 0

foreach ($model in @($spec.models)) {
    $kind = ([string]$model.kind).ToLower()
    if ($kind -notin @("yolo", "anomalib")) {
        throw "模型 $($model.id) 的 kind 只能是 yolo / anomalib，实际是：$kind"
    }

    $sourceDir = Join-Path $repoRoot ([string]$model.source_dir)
    if (-not (Test-Path -LiteralPath $sourceDir)) {
        throw "示例模型的源目录不存在：$sourceDir（改 packaging\models_manifest.json 里的 source_dir）"
    }

    $files = New-Object System.Collections.ArrayList
    $runName = [string]$model.target_run
    if (-not $runName) { throw "模型 $($model.id) 缺少 target_run" }

    if ($kind -eq "yolo") {
        # 安装目录布局：launchers\runs\<run>\weights\<onnx> + labels_<run>.json + args.yaml
        # labels 必须叫 labels_<run>.json：models.py 优先按"引擎所在 run 目录名"精确匹配，
        # 名字对不上时会退化成"取第一份"，一个 run 里有多份标签就会挑错。
        $targetPrefix = "launchers/runs/$runName"
        $weightName = Split-Path -Leaf ([string]$model.source_weight)
        if (-not $weightName.ToLower().EndsWith(".onnx")) {
            throw "YOLO 示例模型必须是 .onnx（引擎绑显卡，不能跨机器发）：$weightName"
        }
        if (-not $weightName.ToLower().Contains(".transd.")) {
            throw "YOLO 示例模型必须用已转置的 *.transd.onnx，否则编出的引擎解码会越界：$weightName"
        }
        Copy-FileRecord -Source (Join-Path $sourceDir ([string]$model.source_weight)) `
            -DestRoot $payload -Relative "$targetPrefix/weights/$weightName" -Sink $files

        $labelsSource = Join-Path $sourceDir ([string]$model.source_labels)
        if (Test-Path -LiteralPath $labelsSource) {
            Copy-FileRecord -Source $labelsSource -DestRoot $payload `
                -Relative "$targetPrefix/labels_$runName.json" -Sink $files
            # 标签文件里的 name 是训练时的实验名（detect_v1）。装到 pcb_demo 下之后
            # 这个名字会显示在推理页的「标签集」上，跟目录名对不上容易让人以为选错文件。
            # classes 原样保留，只把 name 改成目标实验名。
            $labelsPath = Join-Path $payload "$targetPrefix\labels_$runName.json"
            try {
                $labelsJson = Get-Content -LiteralPath $labelsPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $labelsJson | Add-Member -NotePropertyName "name" -NotePropertyValue $runName -Force
                [System.IO.File]::WriteAllText(
                    $labelsPath,
                    ($labelsJson | ConvertTo-Json -Depth 6),
                    (New-Object System.Text.UTF8Encoding $false)
                )
                $labelsItem = Get-Item -LiteralPath $labelsPath
                foreach ($record in $files) {
                    if ($record.path -like "*labels_$runName.json") {
                        $record.bytes = [int64]$labelsItem.Length
                        $record.sha256 = (Get-FileHash -LiteralPath $labelsPath -Algorithm SHA256).Hash
                    }
                }
            } catch {
                Write-Host "  [WARN] 标签文件的 name 没改写成功：$($_.Exception.Message)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  [WARN] 找不到标签文件：$labelsSource（推理时类别只显示编号）"
        }

        # args.yaml 由脚本生成，不直接拷训练目录里的那份：
        # 那份里有开发机的绝对路径（data: D:\...），发给别人既没用还泄露内部目录。
        $argsLines = @()
        if ($model.args) {
            foreach ($prop in $model.args.PSObject.Properties) {
                $argsLines += ("{0}: {1}" -f $prop.Name, $prop.Value)
            }
        } else {
            $argsLines += "task: detect"
        }
        $argsPath = Join-Path $payload "$targetPrefix\args.yaml"
        $argsDir = Split-Path -Parent $argsPath
        if (-not (Test-Path -LiteralPath $argsDir)) { New-Item -ItemType Directory -Path $argsDir -Force | Out-Null }
        [System.IO.File]::WriteAllLines($argsPath, $argsLines, (New-Object System.Text.UTF8Encoding $false))
        $argsItem = Get-Item -LiteralPath $argsPath
        [void]$files.Add([pscustomobject]@{
            path = "$targetPrefix\args.yaml".Replace("/", "\")
            bytes = [int64]$argsItem.Length
            sha256 = (Get-FileHash -LiteralPath $argsPath -Algorithm SHA256).Hash
        })

        $targetHint = "launchers\runs\$runName\weights\$weightName"
    } else {
        # 安装目录布局：launchers\results\<Algo>\<run>\v0\weights\lightning\
        # models.py 从 results 往下最多找 6 层，这个深度是 5 层，正好在范围内。
        $algo = [string]$model.target_algo
        if (-not $algo) { throw "anomalib 模型 $($model.id) 缺少 target_algo" }
        $targetPrefix = "launchers/results/$algo/$runName/v0/weights/lightning"

        $weightName = Split-Path -Leaf ([string]$model.source_weight)
        if (-not $weightName.ToLower().EndsWith(".onnx")) {
            throw "anomalib 示例模型必须是 .onnx：$weightName"
        }
        Copy-FileRecord -Source (Join-Path $sourceDir ([string]$model.source_weight)) `
            -DestRoot $payload -Relative "$targetPrefix/$weightName" -Sink $files

        # metadata.json 必须与引擎同目录：anomalib_infer.dll 从它读阈值与输入尺寸。
        # 但其中 onnx_path / ckpt_path 是开发机的绝对路径，直接把 D:\test\deeplearning\...
        # 发给别人既失效又不专业。这里改成安装目录内的相对路径，
        # 安装脚本落位时会再换成真实绝对路径（降级推理要用它）。
        $metaSource = Join-Path $sourceDir ([string]$model.source_metadata)
        if (-not (Test-Path -LiteralPath $metaSource)) {
            throw "anomalib 示例模型缺少 metadata.json：$metaSource"
        }
        $meta = Get-Content -LiteralPath $metaSource -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $meta.input_size) { throw "$metaSource 缺少 input_size，DLL 会拒绝加载" }
        if (-not $meta.image_threshold -or -not $meta.pixel_threshold) {
            throw "$metaSource 缺少 image_threshold / pixel_threshold"
        }
        $meta | Add-Member -NotePropertyName "ckpt_path" -NotePropertyValue "" -Force
        $meta | Add-Member -NotePropertyName "onnx_path" `
            -NotePropertyValue ("$targetPrefix/$weightName") -Force
        $meta | Add-Member -NotePropertyName "packaged_by" -NotePropertyValue "package_demo_models.ps1" -Force
        $metaPath = Join-Path $payload "$targetPrefix\metadata.json"
        $metaDir = Split-Path -Parent $metaPath
        if (-not (Test-Path -LiteralPath $metaDir)) { New-Item -ItemType Directory -Path $metaDir -Force | Out-Null }
        $metaJson = $meta | ConvertTo-Json -Depth 8
        [System.IO.File]::WriteAllText($metaPath, $metaJson, (New-Object System.Text.UTF8Encoding $false))
        $metaItem = Get-Item -LiteralPath $metaPath
        [void]$files.Add([pscustomobject]@{
            path = "$targetPrefix\metadata.json".Replace("/", "\")
            bytes = [int64]$metaItem.Length
            sha256 = (Get-FileHash -LiteralPath $metaPath -Algorithm SHA256).Hash
        })

        $targetHint = "launchers\results\$algo\$runName\v0\weights\lightning\$weightName"
    }

    $modelBytes = 0
    foreach ($item in $files) { $modelBytes += [int64]$item.bytes }
    $totalBytes += $modelBytes

    [void]$entries.Add([pscustomobject]@{
        id = [string]$model.id
        kind = $kind
        title = [string]$model.title
        task = [string]$model.task
        note = [string]$model.note
        target = $targetHint
        bytes = $modelBytes
        files = $files
    })
}

# ---------------------------------------------------------------------------
# 自检：保证包里没有会失效的东西
# ---------------------------------------------------------------------------
$strayEngines = Get-ChildItem -LiteralPath $payload -Recurse -File |
    Where-Object { $_.Extension -in @(".engine", ".trt") }
if ($strayEngines) {
    # 引擎绑显卡，发出去在别人机器上就是坏文件；宁可打包失败也不要交付"看起来能用"的东西。
    throw "示例模型包里不允许出现 .engine（引擎绑显卡）：$($strayEngines[0].FullName)"
}
$sourceLeaks = Get-ChildItem -LiteralPath $payload -Recurse -File |
    Where-Object { $_.Extension -in @(".py", ".pyc", ".pyd") }
if ($sourceLeaks) { throw "示例模型包里混进了程序文件：$($sourceLeaks[0].FullName)" }

# 密钥扫描：模型包里有 args.yaml / metadata.json / manifest.json 这些文本，
# 而 metadata.json 来自训练目录、里面的字段由训练脚本写。
# 万一训练脚本把带 Key 的配置一起写进 metadata（例如接了大模型做数据增强），
# 这里必须拦下来。见 packaging\secret_scan.ps1。
Assert-NoPackagedSecrets -Root $dest -Label "示例模型包"

# ---------------------------------------------------------------------------
# 清单 + 安装脚本 + 说明
# ---------------------------------------------------------------------------
$manifest = [pscustomobject]@{
    schema = "trtforge.demo_models_package.v1"
    app_version = $appVersion
    packaged_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    total_bytes = $totalBytes
    engine_included = $false
    notes = @(
        "只有 ONNX，没有 .engine：TensorRT 引擎绑显卡型号与 TRT 版本，必须在目标机器上编一次。",
        "装好后打开软件 → 引擎构建页 → 选下面列出的 ONNX → 输出引擎留空 → 开始构建。",
        "编译出的引擎会落在同一个 weights 目录，推理页的下拉列表随即出现这个模型。"
    )
    models = $entries
}
$manifestJson = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
    (Join-Path $dest "manifest.json"),
    $manifestJson,
    (New-Object System.Text.UTF8Encoding $false)
)

$templateDir = Join-Path $scriptDir "templates"
Copy-Item -LiteralPath (Join-Path $templateDir "install_demo_models.bat") `
    -Destination (Join-Path $dest "install_demo_models.bat") -Force
# .ps1 必须带 BOM：PowerShell 5.1 会把没有 BOM 的中文脚本按系统 ANSI 读，直接语法错误
$installText = [System.IO.File]::ReadAllText((Join-Path $templateDir "install_demo_models.ps1"))
[System.IO.File]::WriteAllText(
    (Join-Path $dest "install_demo_models.ps1"),
    $installText,
    (New-Object System.Text.UTF8Encoding $true)
)

# 说明文档按清单现场生成：手写一份的话，改清单忘了改文档就会互相矛盾。
# 下面统一用单引号字符串：说明里有很多反引号（Markdown 代码标记），
# PowerShell 双引号里反引号是转义符，会把 ``` 和 `xxx` 吃掉。
$readme = New-Object System.Collections.ArrayList
[void]$readme.Add('# 示例模型包（配合 TRTForgeWeb 软件包使用）')
[void]$readme.Add('')
[void]$readme.Add(('软件版本：{0}　　打包时间：{1}' -f $appVersion, (Get-Date -Format 'yyyy-MM-dd HH:mm')))
[void]$readme.Add('')
[void]$readme.Add('这个包**不是**软件本体。请先装好软件包 `TRTForgeWeb`，再用这里的模型。')
[void]$readme.Add('')
[void]$readme.Add('## 怎么装')
[void]$readme.Add('')
[void]$readme.Add('1. 把这个文件夹解压到软件目录**旁边**，例如：')
[void]$readme.Add('')
[void]$readme.Add('   ```')
[void]$readme.Add('   D:\TRTForgeWeb\               ← 软件目录（里面有 app\ 和 start_web.bat）')
[void]$readme.Add('   D:\TRTForgeWeb_models_demo\   ← 本文件夹')
[void]$readme.Add('   ```')
[void]$readme.Add('')
[void]$readme.Add('2. 双击 `install_demo_models.bat`。脚本会自动找到软件目录，把模型按正确结构拷进去，')
[void]$readme.Add('   并逐个校验 SHA256；找不到就问你一句路径。')
[void]$readme.Add('')
[void]$readme.Add('3. 双击软件的 `app\TRTForgeWeb.exe`（不需要 PowerShell），打开「引擎构建」页，把下面的 ONNX 编成引擎')
[void]$readme.Add('   （输出引擎**留空**即可，会自动落在同一个 weights 目录）。')
[void]$readme.Add('')
[void]$readme.Add('## 包里有什么')
[void]$readme.Add('')
[void]$readme.Add('| 模型 | 类型 | 装到软件目录的哪里 | 大小 |')
[void]$readme.Add('| --- | --- | --- | --- |')
foreach ($entry in $entries) {
    [void]$readme.Add(('| {0} | {1} | `{2}` | {3:N1} MB |' -f $entry.title, $entry.kind, $entry.target, ($entry.bytes / 1MB)))
}
[void]$readme.Add('')
[void]$readme.Add(('合计约 **{0:N1} MB**。' -f ($totalBytes / 1MB)))
[void]$readme.Add('')
[void]$readme.Add('## 为什么只有 ONNX，没有 .engine')
[void]$readme.Add('')
[void]$readme.Add('`.engine` 是 TensorRT 针对**特定显卡型号 + TRT 版本**序列化出来的，换台机器会报')
[void]$readme.Add('「serialization header」之类的错。只有 `.onnx` 是跨机器通用的，所以到手后要编一次：')
[void]$readme.Add('')
[void]$readme.Add('| 模型 | 编一次的大致耗时 | 显存建议 |')
[void]$readme.Add('| --- | --- | --- |')
[void]$readme.Add('| PCB 检测（960×960，FP16） | 约 8 分钟 | ≥ 4 GB |')
[void]$readme.Add('| EfficientAd（256×256，FP16） | 1～2 分钟 | ≥ 2 GB |')
[void]$readme.Add('')
[void]$readme.Add('编好后会同时生成 `<引擎>.meta.json`（来源记录）和 `<引擎>.engine`，')
[void]$readme.Add('推理页的模型下拉里直接就能选到，标签与阈值都已经在旁边摆好了。')
[void]$readme.Add('')
[void]$readme.Add('## 每个模型的用途')
[void]$readme.Add('')
foreach ($entry in $entries) {
    if ($entry.note) { [void]$readme.Add(('- **{0}**：{1}' -f $entry.title, $entry.note)) }
}
[void]$readme.Add('')
[void]$readme.Add('## 卸载')
[void]$readme.Add('')
[void]$readme.Add('删掉软件目录下 `launchers\runs\<示例名>` 与 `launchers\results\<算法>\<示例名>` 即可，')
[void]$readme.Add('软件不受影响。截图和训练日志都不在这个包里。')
[System.IO.File]::WriteAllLines(
    (Join-Path $dest "README_示例模型.md"),
    $readme,
    (New-Object System.Text.UTF8Encoding $false)
)

if (-not $SkipZip) {
    $zip = Join-Path $OutDir "TRTForgeWeb_models_demo.zip"
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Push-Location $OutDir
    try {
        tar -a -c -f $zip "TRTForgeWeb_models_demo"
        if ($LASTEXITCODE -ne 0) { throw "压缩失败" }
    } finally {
        Pop-Location
    }
    $zipItem = Get-Item -LiteralPath $zip
    Write-Host ("压缩包: {0}（{1:N1} MB）" -f $zip, ($zipItem.Length / 1MB))
}

Write-Host ""
Write-Host ("示例模型包: {0}" -f $dest) -ForegroundColor Green
Write-Host ("模型数: {0}　解压后 {1:N1} MB　只有 ONNX" -f $entries.Count, ($totalBytes / 1MB))
Write-Host "用法：解压到软件目录旁边，双击 install_demo_models.bat，再到「引擎构建」页编一次引擎。"
