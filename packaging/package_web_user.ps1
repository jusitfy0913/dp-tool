# -*- coding: utf-8 -*-
# 打「推理 + 训练」用户包。对方电脑统一为 CUDA 11.8。
#
# 包里有：编译后的网页程序、编译后的训练启动器、推理 DLL、
# CUDA 11.8 / TensorRT / OpenCV 运行库、网页文件、config.json。
# 包里没有：C++ 源码、网页后端 .py、YOLO/Anomalib 源码树、Qt 程序，**也不含示例模型**。
#
# 示例模型是另一个包（packaging\package_demo_models.ps1），理由：
#   - 模型换得比软件勤，塞在一起会让每次换模型都重打 1.8 GB；
#   - 示例模型只带 ONNX（引擎绑显卡），需要在目标机器上编一次，生命周期和软件不同；
#   - 只要软件不要模型的人，下载时间能少一截。
# 两个包解压到同一层目录，双击模型包里的 install_demo_models.bat 即可合体。
#
# 在 dp-tool 目录执行：
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\package_web_user.ps1
#
# 可选：
#   -OutDir D:\dist
#   -SkipZip
#   -WithDemoModels          把示例模型一起打进来（默认不带：模型单独出一个包，
#                            包里只有 ONNX，见 packaging\package_demo_models.ps1）
#   -DemoModelsDir <目录>    已解压的示例模型包目录（默认 <OutDir>\TRTForgeWeb_models_demo）

param(
    [string]$OutDir = "",
    [switch]$SkipZip,
    [switch]$WithOcr,
    [string]$OcrModelDir = "",
    [switch]$WithDemoModels,
    [string]$DemoModelsDir = "",
    # 训练库 wheel 的缓存目录。默认 <OutDir>\_wheels_cache：
    # 一份 3 GB 的 wheel 集，每次打包都重新下载既慢又容易触发镜像限流。
    [string]$WheelsDir = ""
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
# 密钥扫描器：打包最后一步必须过这一关。见 packaging\secret_scan.ps1 顶部说明。
. (Join-Path $scriptDir "secret_scan.ps1")

$repoRoot = Find-RepoRoot -StartDir $scriptDir
if (-not $repoRoot) { throw "找不到仓库根目录（需要 6-web 和 4-qt-studio）。" }

$python = Find-Python
$binRoot = Join-Path $repoRoot "4-qt-studio\bin"
$backend = Join-Path $repoRoot "6-web\backend"
$frontend = Join-Path $repoRoot "6-web\frontend"

# 窗口模式（WebView2）的依赖必须在**编译机**上装好，它们会被编进 exe。
# 这两个包不放 wheels\：对方机器不能靠 pip 装它们，否则又回到「联网才能用」。
# 宁可在这里停住，也不要编出一个双击打不开的 exe。
$uiProbe = & $python -c "import importlib.util as u; print('webview=' + str(u.find_spec('webview') is not None), 'pythonnet=' + str(u.find_spec('pythonnet') is not None), 'clr_loader=' + str(u.find_spec('clr_loader') is not None))"
if ($LASTEXITCODE -ne 0 -or $uiProbe -notmatch "webview=True" -or $uiProbe -notmatch "pythonnet=True" -or $uiProbe -notmatch "clr_loader=True") {
    throw "编译机缺少 pywebview / pythonnet / clr_loader（$uiProbe）。请先执行：& `"$python`" -m pip install pywebview==6.2.1 pythonnet"
}
Write-Host "  窗口模式依赖就绪：$uiProbe"

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
# 示例模型的落点。软件包自己不带模型（示例模型单独一个包），
# 但这两个目录必须预先建好，否则用户把示例模型包解压进来时会被多套一层目录，
# 后端扫不到（models.py 只认 launchers\runs 与 launchers\results）。
New-Item -ItemType Directory -Path (Join-Path $dest "launchers\runs") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dest "launchers\results") | Out-Null
New-Item -ItemType Directory -Path $build | Out-Null

Write-Host "[1/7] 复制推理运行库（CUDA 11.8 / TensorRT / ONNX Runtime / OpenCV）"
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

Write-Host "[2/7] 复制网页界面"
& robocopy $frontend (Join-Path $dest "frontend") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "复制前端失败，robocopy 退出码 $LASTEXITCODE" }

# ---------------------------------------------------------------------------
# 离线依赖包：对方电脑不再需要联网 pip install 才能训练
#
# 为什么必须自带：训练/建库要在目标机器上真的 import torch、timm、ultralytics。
# 现场没有外网时「先联网装库」这条路直接走不通；而且同一份代码在不同时间联网装，
# 拿到的版本可能不同，出了问题无法复现。
#
# 做法是一个 wheelhouse 目录 + 安装脚本里 `pip install --no-index --find-links`：
# torch 的 cu118 wheel 自带 CUDA 运行库，不需要额外装 cuDNN。
# ---------------------------------------------------------------------------
Write-Host "[3/7] 下载离线依赖包（训练库 wheel）"
if (-not $WheelsDir) { $WheelsDir = Join-Path $OutDir "_wheels_cache" }
New-Item -ItemType Directory -Path $WheelsDir -Force | Out-Null

$torchIndex = "https://download.pytorch.org/whl/cu118"
# 版本必须与 templates\安装训练环境.ps1 完全一致，否则现场环境与开发机不同。
# polars 与 polars-runtime-compat 必须同版本：后者是前者去掉 AVX2 依赖的重编版。
#
# 注意 wheels 里还会多出一份 polars-runtime-32（polars 与 ultralytics 声明的依赖）。
# 实测删掉它 pip 就报「找不到 polars-runtime-32」，离线装 ultralytics 直接失败，
# 所以它必须留在包内；由安装脚本装完立刻卸掉，最终环境只有 compat 版。
$polarsVersion = "1.36.1"
#: 带 +cu118 本地版本的 torch：只能从 PyTorch 官方源拿到，PyPI 上的是 CPU 版。
$torchRequirements = @(
    "torch==2.3.0+cu118",
    "torchvision==0.18.0+cu118"
)
$wheelRequirements = @(
    "torch==2.3.0+cu118",
    "torchvision==0.18.0+cu118",
    "ultralytics==8.4.192",
    "anomalib==1.2.0",
    "timm==1.0.14",
    # 这三个钉死成开发机实测过的版本。不钉会装到最新版：实测一次打出来的包是
    # opencv-python 5.0 / onnx 1.23 / onnxruntime 1.32，而开发机跑的是 4.11 / 1.16 / 1.20。
    # 现场装上一套没人验证过的 OpenCV 5，出了问题只能远程猜。
    "onnx==1.16.1",
    "onnxruntime==1.20.1",
    "opencv-python==4.11.0.86",
    "polars==$polarsVersion",
    "polars-runtime-compat==$polarsVersion",
    # pip 的 wheel 是给「离线创建 Python 环境」用的：嵌入式发行版不带 pip，
    # 安装脚本直接把这个 wheel 解压到 site-packages 就能引导，不用联网跑 get-pip.py。
    "pip"
)
# 必须能在包里找到的发行版（按 wheel 文件名前缀判断，下载后逐个核对）
$requiredWheels = @(
    "torch", "torchvision", "ultralytics", "anomalib", "timm",
    "onnx", "onnxruntime", "polars", "polars_runtime_compat",
    "numpy", "opencv_python", "pip"
)
function Get-WheelCount {
    param([string]$Dir, [string]$Name)
    @(Get-ChildItem -LiteralPath $Dir -Filter "$Name-*.whl" -File -ErrorAction SilentlyContinue).Count
}
$missingWheels = @($requiredWheels | Where-Object { (Get-WheelCount -Dir $WheelsDir -Name $_) -eq 0 })
if ($missingWheels.Count -gt 0) {
    Write-Host "  缓存里缺 $($missingWheels.Count) 个库，开始下载（约 3 GB，第一次比较久）"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    # 分两步下载。为什么不一次 `pip download` 加 extra-index：解析器会为了每个包都去
    # 问一遍 PyTorch 的索引，实测 5 分钟都走不完。先单独拿 cu118 的 torch（--no-deps，
    # 它的依赖都在 PyPI），再让第二步用 --find-links 认本地这份，速度快一个量级。
    $torchReqFile = Join-Path $build "wheels-torch.txt"
    [System.IO.File]::WriteAllLines($torchReqFile, $torchRequirements, $utf8NoBom)
    # 官方 CDN 在部分网络下会返回一个很短的错误页，pip 把它当"哈希不符"（实测连续两次）。
    # 备用源是 PyTorch wheel 的国内镜像：同名同版本，下完仍按官方 sha256 核对。
    # 上海交大实测 4 MB/s，阿里云只有 0.5 MB/s，所以交大排前面。
    $torchSources = @(
        @{ Label = "PyTorch 官方源"; Args = @("--index-url", $torchIndex) },
        @{ Label = "上海交大镜像"; Args = @("--find-links", "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/") },
        @{ Label = "阿里云镜像"; Args = @("--find-links", "https://mirrors.aliyun.com/pytorch-wheels/cu118/") }
    )
    $torchOk = $false
    foreach ($source in $torchSources) {
        Write-Host "  从$($source.Label)下载 cu118 版 PyTorch"
        & $python -m pip download --only-binary=:all: --platform win_amd64 --python-version 3.12 `
            --implementation cp --abi cp312 --no-deps --retries 3 --dest $WheelsDir @($source.Args) -r $torchReqFile
        if ($LASTEXITCODE -eq 0) { $torchOk = $true; break }
        Write-Host "  $($source.Label)没成功，换下一个源。"
    }
    if (-not $torchOk) { throw "下载 cu118 版 PyTorch 失败。检查网络后重试。" }
    # 镜像可能换过文件：这里按官方索引公布的 sha256 逐个核对，不一致就停止打包。
    $torchHashes = @{
        "torch-2.3.0+cu118-cp312-cp312-win_amd64.whl"       = "9c396ea823d5275b806045e3e66a9c2e8382c9fe83a2bcaac1c2bd0f0e45bff4"
        "torchvision-0.18.0+cu118-cp312-cp312-win_amd64.whl" = "6c8799bce134c1abe1dc846ea829ef05279e07b60b51fcbcfd6a3fc3be0bf790"
    }
    foreach ($pair in $torchHashes.GetEnumerator()) {
        $wheelPath = Join-Path $WheelsDir $pair.Key
        if (-not (Test-Path -LiteralPath $wheelPath)) { throw "没有下到 $($pair.Key)" }
        $actualHash = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLower()
        if ($actualHash -ne $pair.Value) {
            throw "$($pair.Key) 的 sha256 与官方不一致（$actualHash），拒绝打包。"
        }
    }

    $reqFile = Join-Path $build "wheels-requirements.txt"
    [System.IO.File]::WriteAllLines($reqFile, $wheelRequirements, $utf8NoBom)
    & $python -m pip download --only-binary=:all: --platform win_amd64 --python-version 3.12 `
        --implementation cp --abi cp312 --dest $WheelsDir --find-links $WheelsDir -r $reqFile
    if ($LASTEXITCODE -ne 0) { throw "下载离线依赖包失败。检查网络，或用 -WheelsDir 指向已下好的目录。" }
} else {
    Write-Host "  复用缓存：$WheelsDir"
}

# 缺库就停：带出去的包装不上训练环境 = 用户拿到一个只能推理的壳。
$stillMissing = @($requiredWheels | Where-Object { (Get-WheelCount -Dir $WheelsDir -Name $_) -eq 0 })
if ($stillMissing.Count -gt 0) { throw "离线依赖包不全，缺少：$($stillMissing -join ', ')。已停止打包。" }

& robocopy $WheelsDir (Join-Path $dest "wheels") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "复制 wheel 目录失败，robocopy 退出码 $LASTEXITCODE" }
$wheelLines = @(Get-ChildItem -LiteralPath (Join-Path $dest "wheels") -Filter *.whl -File | Sort-Object Name | ForEach-Object { $_.Name })
[System.IO.File]::WriteAllLines((Join-Path $dest "wheels\manifest.txt"), $wheelLines, (New-Object System.Text.UTF8Encoding $false))
$wheelSize = [math]::Round((Get-ChildItem -LiteralPath (Join-Path $dest "wheels") -File | Measure-Object Length -Sum).Sum / 1GB, 2)
Write-Host "  已带上 $($wheelLines.Count) 个 wheel，$wheelSize GB"

# 嵌入式 Python：对方机器没有 Anaconda、又不能联网时，靠它建训练环境。
# 11 MB，代价很小；缺了它，离线安装就只能装库、建不了 Python 环境。
$runtimeDir = Join-Path $dest "runtime"
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$embedZip = Join-Path $runtimeDir "python-embed.zip"
if (-not (Test-Path -LiteralPath $embedZip)) {
    $embedUrls = @(
        "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip",
        "https://registry.npmmirror.com/-/binary/python/3.12.10/python-3.12.10-embed-amd64.zip"
    )
    $gotEmbed = $false
    foreach ($url in $embedUrls) {
        try {
            Write-Host "  下载嵌入式 Python 3.12：$url"
            Invoke-WebRequest -Uri $url -OutFile "$embedZip.part" -UseBasicParsing -TimeoutSec 300
            if ((Get-Item -LiteralPath "$embedZip.part").Length -lt 5000000) { throw "文件太小，不像是完整安装包" }
            Move-Item -LiteralPath "$embedZip.part" -Destination $embedZip -Force
            $gotEmbed = $true
            break
        } catch {
            Write-Host "  这个源不行：$($_.Exception.Message)"
            if (Test-Path -LiteralPath "$embedZip.part") { Remove-Item -LiteralPath "$embedZip.part" -Force }
        }
    }
    if (-not $gotEmbed) { throw "下载嵌入式 Python 失败，离线安装会缺 Python 环境。已停止打包。" }
}
Write-Host "  已带上嵌入式 Python：$([math]::Round((Get-Item -LiteralPath $embedZip).Length / 1MB, 1)) MB"

Write-Host "[4/7] 编译训练启动器（.pyd，不含 torch 本体）"
& $python -m pip install -q nuitka ordered-set zstandard
if ($LASTEXITCODE -ne 0) { throw "安装 Nuitka 失败" }

# 这些库不编进 .pyd，而是由 wheels\ 里的 wheel 在对方机器上离线装上。
# 两者缺一不可：只 nofollow 不给 wheel，用户 import 时就 ModuleNotFound（这正是
# 「安装包必须自带依赖库」那条规矩要防的事）；只给 wheel 不 nofollow，Nuitka 会把
# 2.6 GB 的 torch 冻进 .pyd，编译几小时且装完必崩。
$nofollow = @(
    "torch", "torchvision", "ultralytics", "anomalib",
    "lightning", "pytorch_lightning", "cv2", "onnx", "onnxruntime", "timm"
)
function Compile-Launcher {
    param([string]$SourcePy, [string]$ModuleName, [string[]]$ExtraSources = @())
    $work = Join-Path $build $ModuleName
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    $moduleFile = Join-Path $work "$ModuleName.py"
    Copy-Item -LiteralPath $SourcePy -Destination $moduleFile -Force
    # 同目录的辅助模块必须一起放进编译目录，Nuitka 才会把它编进同一个 .pyd。
    # 漏了的话打包不报错，用户点「开始」才 ModuleNotFoundError。
    $extraNames = @()
    foreach ($extra in $ExtraSources) {
        Copy-Item -LiteralPath $extra -Destination (Join-Path $work (Split-Path -Leaf $extra)) -Force
        $extraNames += [System.IO.Path]::GetFileNameWithoutExtension($extra)
    }
    $nuitkaArgs = @(
        "-m", "nuitka", "--module", $moduleFile,
        "--output-dir=$work",
        "--remove-output",
        "--assume-yes-for-downloads"
    )
    foreach ($name in $nofollow) { $nuitkaArgs += "--nofollow-import-to=$name" }
    # `--module` 模式不会自动跟随「编译目录里的同级 .py」：光 Copy-Item 过去，
    # .pyd 能编出来，但运行到 `import anomalydino_core` 那行才 ModuleNotFoundError。
    # 所以辅助模块必须显式 --include-module，并且要 cd 到编译目录让它在 sys.path 上。
    foreach ($name in $extraNames) { $nuitkaArgs += "--include-module=$name" }
    $reportFile = Join-Path $work "nuitka-report.xml"
    $nuitkaArgs += "--report=$reportFile"
    Push-Location -LiteralPath $work
    try {
        & $python @nuitkaArgs
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) { throw "编译失败：$SourcePy" }
    # 光看退出码不够：辅助模块没编进去也能编出 .pyd。这里按 report.xml 逐个数。
    if ($extraNames.Count -gt 0) {
        $reportText = Get-Content -LiteralPath $reportFile -Raw -Encoding UTF8
        foreach ($name in $extraNames) {
            if ($reportText -notmatch ('<module name="' + [regex]::Escape($name) + '"')) {
                throw "编译产物缺少辅助模块 $name（运行时才会 ModuleNotFoundError）：$ModuleName"
            }
        }
    }
    $pyd = Get-ChildItem -LiteralPath $work -Filter "$ModuleName*.pyd" | Select-Object -First 1
    if (-not $pyd) { throw "没有生成 $ModuleName.pyd" }
    Copy-Item -LiteralPath $pyd.FullName -Destination (Join-Path $dest "launchers\$($pyd.Name)") -Force
}

Compile-Launcher -SourcePy (Join-Path $repoRoot "1-python-yolo\launcher\tf_yolo_launcher.py") -ModuleName "tf_yolo_launcher_bin"
Compile-Launcher -SourcePy (Join-Path $repoRoot "2-python-anomaly\launcher\tf_anomalib_launcher.py") -ModuleName "tf_anomalib_launcher_bin"
# AnomalyDINO 的算法在 anomalydino_core.py 里，必须和启动器一起编进同一个 .pyd
Compile-Launcher -SourcePy (Join-Path $repoRoot "2-python-anomaly\launcher\tf_anomalydino_launcher.py") -ModuleName "tf_anomalydino_launcher_bin" -ExtraSources @(
    (Join-Path $repoRoot "2-python-anomaly\launcher\anomalydino_core.py")
)

$stub = @"
# -*- coding: utf-8 -*-
from __future__ import annotations

import {0} as impl

raise SystemExit(impl.main())
"@
$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Join-Path $dest "launchers\tf_yolo_launcher.py"), ($stub -f "tf_yolo_launcher_bin"), $utf8)
[System.IO.File]::WriteAllText((Join-Path $dest "launchers\tf_anomalib_launcher.py"), ($stub -f "tf_anomalib_launcher_bin"), $utf8)
[System.IO.File]::WriteAllText((Join-Path $dest "launchers\tf_anomalydino_launcher.py"), ($stub -f "tf_anomalydino_launcher_bin"), $utf8)

Write-Host "[5/7] 编译网页程序（这一步较久）"
$serverOut = Join-Path $build "server"
New-Item -ItemType Directory -Path $serverOut -Force | Out-Null
$serverArgs = @(
    "-m", "nuitka", "--standalone",
    "--output-dir=$serverOut",
    "--output-filename=TRTForgeWeb.exe",
    "--assume-yes-for-downloads",
    "--enable-plugin=tk-inter",
    "--windows-console-mode=attach",
    "--include-module=launcher",
    # 桌面窗口（WebView2）。desktop.py 是 launcher 里**延迟 import** 的
    # （ui=browser 的用户不必付 pythonnet 的 import 成本），静态分析不一定跟得上，
    # 这里显式钉住。
    #
    # 注意不要写 --include-package=webview：Nuitka 自带 pywebview 插件（`--plugin-list`
    # 里能看到 "pywebview  Required by 'webview' package"），插件会自己决定哪些
    # 平台模块进来；手动 include 会和它的排除决定打架，实测报
    # "Conflict between user and plugin decision for module 'webview.platforms.android'"
    # 然后直接 FATAL。让插件干活即可。
    "--include-module=desktop",
    # 路由按域拆在 routers/ 包里（v1.38.0）。逐个列子模块会漏，包名一次覆盖全部。
    "--include-package=routers",
    "--include-module=web_state",
    "--include-module=tools.val_engine",
    "--include-module=tools.val_anomaly",
    "--include-module=tools.infer_batch",
    "--include-module=tools.infer_video",
    "--include-module=infer_worker",
    "--include-module=tools.int8_launch",
    "--include-module=int8calib",
    "--include-module=inferproxy",
    "--include-module=uvicorn.logging",
    "--include-module=uvicorn.loops",
    "--include-module=uvicorn.loops.auto",
    "--include-module=uvicorn.protocols.http.auto",
    "--include-module=uvicorn.protocols.websockets.auto",
    "--include-module=uvicorn.lifespan.on",
    # 路由模块到底进没进包，靠这份报告核对（不支持「看有没有 routers 目录」，
    # standalone 会把纯 Python 模块链进 exe）。
    "--report=$(Join-Path $serverOut 'nuitka-report.xml')",
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
# 路由包必须真进包：漏了 `routers`，exe 一起来就是 404，而且只在安装形态下炸。
#
# 不能靠「产物里有没有 routers 目录」判断：Nuitka 的 standalone 会把编译后的纯 Python
# 模块**链进 exe**（实测 v1.40.0：dist 里没有 routers 目录，但 101 MB 的 exe 里全都有）。
# 所以改成读 Nuitka 的 --report：路由目录下每个 .py 都必须作为模块出现在报告里。
$serverReport = Join-Path $serverOut "nuitka-report.xml"
if (-not (Test-Path -LiteralPath $serverReport)) { throw "缺少编译报告：$serverReport" }
$reportText = Get-Content -LiteralPath $serverReport -Raw -Encoding UTF8
$routerMissing = @()
foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $backend "routers") -Filter "*.py" -File)) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $moduleName = if ($stem -eq "__init__") { "routers" } else { "routers.$stem" }
    if ($reportText -notmatch ('<module name="' + [regex]::Escape($moduleName) + '"')) {
        $routerMissing += $moduleName
    }
}
if ($routerMissing.Count -gt 0) {
    throw "编译结果缺少路由模块：$($routerMissing -join ', ')（检查 Nuitka 的 --include-package=routers）"
}
& robocopy $distDir (Join-Path $dest "app") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "复制编译结果失败" }
$appDir = Join-Path $dest "app"

# ---------------------------------------------------------------------------
# 补 DLL 传递依赖
#
# 教训（v1.40.0）：Conda 装出来的 Pillow 把编译库链到 <env>\Library\bin
# （tiff.dll、openjp2.dll……），Nuitka 只搬了 zlib.dll。结果安装包里
# `import PIL.Image` 抛一个**空消息**的 ImportError，用户双击就闪退；
# 开发机上一切正常，因为开发机 PATH 里有 conda 的 Library\bin。
#
# 所以不能靠手写清单（以前就是手写 libssl/tcl 那几个，早晚漏），
# 改成扫 PE 导入表做**传递闭包**：谁缺就从编译环境里找出来补到 exe 旁边。
# 找不到就当作系统 DLL（kernel32、api-ms-win-*、VC 运行库由 Nuitka 自带）。
function Get-PeImportedDlls {
    param([string]$Path, [string]$Dumpbin)
    $lines = & $Dumpbin /nologo /dependents $Path 2>$null
    if ($LASTEXITCODE -ne 0) { return @() }
    $names = New-Object System.Collections.Generic.List[string]
    $inList = $false
    foreach ($line in $lines) {
        $t = $line.Trim()
        # 空行不能当结束符：dumpbin 在 "following dependencies:" 后面先空一行才列 DLL。
        if ($t -eq "") { continue }
        if ($t -match "following (delay load )?dependencies:") { $inList = $true; continue }
        if (-not $inList) { continue }
        if ($t -match '^[A-Za-z0-9_\.\-]+\.dll$') {
            $names.Add($t)
            continue
        }
        # 走到 "Summary" 或下一节标题就收工。
        $inList = $false
    }
    return $names
}

function Find-Dumpbin {
    # Nuitka 编译靠 cl.exe，所以机器上一定有 MSVC；dumpbin 就躺在 cl.exe 旁边。
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cl) {
        $candidate = Join-Path (Split-Path -Parent $cl.Source) "dumpbin.exe"
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $vsRoot = & $vswhere -latest -products * -property installationPath 2>$null
        if ($vsRoot) {
            $found = Get-ChildItem -LiteralPath (Join-Path $vsRoot.Trim() "VC\Tools\MSVC") -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { Join-Path $_.FullName "bin\Hostx64\x64\dumpbin.exe" } |
                Where-Object { Test-Path -LiteralPath $_ } |
                Select-Object -First 1
            if ($found) { return $found }
        }
    }
    return $null
}

$dumpbin = Find-Dumpbin
if (-not $dumpbin) {
    throw "找不到 dumpbin.exe（MSVC 工具链）。补 DLL 依赖要靠它读 PE 导入表，不能用猜测替代。"
}

$envRoot = Split-Path -Parent $python
# 补库的搜索目录：conda 的 Library\bin 是主战场，其余目录兜底。
$dllSearchDirs = @(
    (Join-Path $envRoot "Library\bin"),
    (Join-Path $envRoot "DLLs"),
    $envRoot
) | Where-Object { Test-Path -LiteralPath $_ }

$appDlls = @{}
# 注意：-Include 配 -LiteralPath 在 PS 5.1 下会被忽略（会把 2600 多个文件全列出来），
# 所以这里手动按扩展名过滤，只扫真正的二进制。
foreach ($file in (Get-ChildItem -LiteralPath $appDir -Recurse -File | Where-Object { $_.Extension -in @(".dll", ".pyd") })) {
    $appDlls[$file.Name.ToLowerInvariant()] = $file.FullName
}

# 系统 DLL 一律不搬，尤其不能搬 conda 的 UCRT 转发件（api-ms-win-crt-*）：
# 它们和系统 UCRT 版本打架，症状是进程一启动就 0xC0000409（abort），
# 而且日志里一个字都不留，比缺库还难查。这些交给系统和 Nuitka 自己带。
function Test-SystemDll {
    param([string]$Name)
    return ($Name -match '^(api-ms-win-|ucrtbase\.dll|msvcrt\.dll|vcruntime|msvcp|concrt|kernel32|user32|gdi32|advapi32|shell32|ole32|oleaut32|ws2_32|wsock32|rpcrt4|bcrypt|crypt32|ntdll|psapi|version\.dll|shlwapi|iphlpapi|secur32|netapi32|userenv)')
}

$copiedDlls = New-Object System.Collections.Generic.List[string]
$unresolved = New-Object System.Collections.Generic.List[string]
$queue = New-Object System.Collections.Generic.Queue[string]
foreach ($name in $appDlls.Keys) { $queue.Enqueue($name) }
$scanned = @{}

while ($queue.Count -gt 0) {
    $current = $queue.Dequeue()
    if ($scanned.ContainsKey($current)) { continue }
    $scanned[$current] = $true
    $source = $appDlls[$current]
    foreach ($dep in (Get-PeImportedDlls -Path $source -Dumpbin $dumpbin)) {
        $depKey = $dep.ToLowerInvariant()
        if ($appDlls.ContainsKey($depKey)) { continue }
        if (Test-SystemDll -Name $dep) { continue }
        $found = $null
        foreach ($dir in $dllSearchDirs) {
            $candidate = Join-Path $dir $dep
            if (Test-Path -LiteralPath $candidate) { $found = $candidate; break }
        }
        if (-not $found) {
            # 系统 DLL 不在安装包里，不追；只记下来便于排查。
            $unresolved.Add($dep)
            continue
        }
        $target = Join-Path $appDir $dep
        Copy-Item -LiteralPath $found -Destination $target -Force
        $appDlls[$depKey] = $target
        $copiedDlls.Add($dep)
        $queue.Enqueue($depKey)
        $scanned.Remove($depKey)
    }
}
if ($copiedDlls.Count -gt 0) {
    Write-Host ("  补齐 DLL 依赖：{0} 个（{1}）" -f $copiedDlls.Count, (($copiedDlls | Sort-Object -Unique) -join ", "))
}

# Tk/Tcl 的数据目录不在 DLL 依赖里，单独搬。
foreach ($pair in @(
    @{ Src = Join-Path $envRoot "Library\lib\tcl8.6"; Dest = Join-Path $appDir "tcl8.6" },
    @{ Src = Join-Path $envRoot "Library\lib\tk8.6"; Dest = Join-Path $appDir "tk8.6" }
)) {
    if (Test-Path -LiteralPath $pair.Src) {
        & robocopy $pair.Src $pair.Dest /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "复制 $($pair.Src) 失败" }
    }
}

# 装完就得当场证明它真能起来：import 全过不去就停在这里，不出 zip。
# 静态检查（文件在不在）已经骗过我们一次，所以这次真跑一遍 exe。
Write-Host "  自检：在安装包里真跑一遍 import"
$selfcheckOut = Join-Path $build "selfcheck.out.txt"
$selfcheckErr = Join-Path $build "selfcheck.err.txt"
$selfcheckExe = Join-Path $appDir "TRTForgeWeb.exe"
foreach ($f in @($selfcheckOut, $selfcheckErr)) {
    if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force }
}
$env:TRTFORGE_HOME = $dest
# 用 Start-Process 而不是 `&`：需要拿**进程退出码**。
# 这个 exe 是 Nuitka 的 attach 模式，崩起来是 0xC0000409，`$LASTEXITCODE` 拿不到；
# 重定向到文件则保证 stdout/stderr 都能留下证据（哪怕进程已崩）。
$selfcheckProc = Start-Process -FilePath $selfcheckExe -ArgumentList "--tool", "selfcheck" `
    -WorkingDirectory $appDir -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $selfcheckOut -RedirectStandardError $selfcheckErr
$selfcheckCode = $selfcheckProc.ExitCode
Remove-Item Env:\TRTFORGE_HOME -ErrorAction SilentlyContinue
foreach ($f in @($selfcheckOut, $selfcheckErr)) {
    if (Test-Path -LiteralPath $f) {
        Get-Content -LiteralPath $f -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }
    }
}
if ($selfcheckCode -ne 0) {
    throw ("安装包自检未通过（退出码 $selfcheckCode），不允许出 zip。日志：$selfcheckOut / $selfcheckErr")
}
# 自检只是 import，正常不落盘。万一某个模块在 import 期建了 runtime 目录，
# 别把这个空目录带进安装包（首次启动时 config.ensure_dirs() 会自己建）。
$strayRuntime = Join-Path $appDir "runtime"
if (Test-Path -LiteralPath $strayRuntime) {
    Remove-Item -LiteralPath $strayRuntime -Recurse -Force
}

Write-Host "[6/7] 写入配置、启动脚本和操作手册"
$templateDir = Join-Path $scriptDir "templates"
Copy-Item -LiteralPath (Join-Path $templateDir "config.json") -Destination (Join-Path $dest "config.json") -Force
Copy-Utf8 -Source (Join-Path $templateDir "start_user.ps1") -Dest (Join-Path $dest "start_user.ps1")
Copy-Item -LiteralPath (Join-Path $templateDir "start_web.bat") -Destination (Join-Path $dest "start_web.bat") -Force
Copy-Item -LiteralPath (Join-Path $templateDir "start_web_cmd.bat") -Destination (Join-Path $dest "start_web_cmd.bat") -Force
Copy-Utf8 -Source (Join-Path $scriptDir "templates\安装训练环境.ps1") -Dest (Join-Path $dest "install_train_env.ps1")
Copy-Item -LiteralPath (Join-Path $templateDir "install_train_env.bat") -Destination (Join-Path $dest "install_train_env.bat") -Force
Copy-Utf8 -Source (Join-Path $scriptDir "templates\一键安装.ps1") -Dest (Join-Path $dest "一键安装.ps1")
Copy-Item -LiteralPath (Join-Path $templateDir "一键安装.bat") -Destination (Join-Path $dest "一键安装.bat") -Force
Copy-Utf8 -Source (Join-Path $scriptDir "使用说明.md") -Dest (Join-Path $dest "使用说明.md")

$manual = Join-Path $repoRoot "docs\TRTForge_Web操作手册.md"
$manualImages = Join-Path $repoRoot "docs\images\trtforge_web_manual"
if (Test-Path -LiteralPath $manual) {
    New-Item -ItemType Directory -Path (Join-Path $dest "docs") -Force | Out-Null
    Copy-Utf8 -Source $manual -Dest (Join-Path $dest "docs\TRTForge_Web操作手册.md")
}
$noPsDoc = Join-Path $repoRoot "docs\启动与排障说明.md"
if (Test-Path -LiteralPath $noPsDoc) {
    New-Item -ItemType Directory -Path (Join-Path $dest "docs") -Force | Out-Null
    Copy-Utf8 -Source $noPsDoc -Dest (Join-Path $dest "docs\启动与排障说明.md")
}
if (Test-Path -LiteralPath $manualImages) {
    & robocopy $manualImages (Join-Path $dest "docs\images\trtforge_web_manual") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
}

# ---------------------------------------------------------------------------
# 示例模型的落点说明
#
# 软件包默认**不带**模型：模型换得比软件勤，合成一个包会让每次换模型都得重打 1.8 GB。
# 但空目录在压缩包里会被丢掉，所以每个目录放一份说明文件当占位；
# 这也顺便告诉用户"模型去哪拿"，避免他以为软件坏了。
# ---------------------------------------------------------------------------
$modelsReadme = @"
示例模型不在这台安装包里。

模型单独打一个包（约 40 MB），里面只有 ONNX，没有 .engine：

    1. 把示例模型包解压到本目录**旁边**（与本目录同级）
    2. 双击示例模型包里的 install_demo_models.bat
    3. 打开软件 →「引擎构建」页 → 选 ONNX → 输出引擎留空 → 开始构建
    4. 回到「推理演示」页，模型下拉里就会出现示例模型

为什么不直接放 .engine：
    TensorRT 引擎绑定显卡型号与 TRT 版本，开发机编好的引擎换台机器会报错，
    必须在对方显卡上编一次（PCB 模型约 8 分钟）。

不打示例模型也能正常使用软件：训练、标注、引擎构建、文字识别都不依赖它，
只是推理页的下拉列表一开始是空的。
"@
[System.IO.File]::WriteAllText((Join-Path $dest "models\README.txt"), $modelsReadme, $utf8)

$runsReadme = @"
本目录放 YOLO 训练产物，结构：runs\<实验名>\weights\<引擎>，标签 labels_<实验名>.json 放在实验根。

示例模型包会装到 runs\pcb_demo\。没有示例模型时这里是空的，不影响使用。
"@
[System.IO.File]::WriteAllText((Join-Path $dest "launchers\runs\README.txt"), $runsReadme, $utf8)

$resultsReadme = @"
本目录放异常检测训练产物，结构：results\<算法>\<实验名>\v0\weights\lightning\<引擎 + metadata.json>。

示例模型包会装到 results\EfficientAd\pcb_demo\。没有示例模型时这里是空的，不影响使用。
"@
[System.IO.File]::WriteAllText((Join-Path $dest "launchers\results\README.txt"), $resultsReadme, $utf8)

if ($WithDemoModels) {
    # 只有明确要求时才把示例模型并进软件包（离线交付、对方不会解压第二个包的场景）。
    if (-not $DemoModelsDir) { $DemoModelsDir = Join-Path $OutDir "TRTForgeWeb_models_demo" }
    $demoPayload = Join-Path $DemoModelsDir "payload"
    if (-not (Test-Path -LiteralPath $demoPayload)) {
        throw "找不到示例模型：$demoPayload。先跑 packaging\package_demo_models.ps1，或去掉 -WithDemoModels。"
    }
    $demoManifest = Join-Path $DemoModelsDir "manifest.json"
    if (-not (Test-Path -LiteralPath $demoManifest)) { throw "示例模型包缺少 manifest.json：$demoManifest" }
    $demoSpec = Get-Content -LiteralPath $demoManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    $demoFiles = 0
    foreach ($model in @($demoSpec.models)) {
        foreach ($file in @($model.files)) {
            $relative = [string]$file.path
            $source = Join-Path $demoPayload $relative
            if (-not (Test-Path -LiteralPath $source)) { throw "示例模型包缺文件：$source" }
            $target = Join-Path $dest $relative
            $targetDir = Split-Path -Parent $target
            if (-not (Test-Path -LiteralPath $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            Copy-Item -LiteralPath $source -Destination $target -Force
            $demoFiles += 1
        }
        # anomalib 的 metadata.json 里 onnx_path 要指向安装目录内的真实位置，
        # 因为换机器后降级推理要靠它找回同源 ONNX。
        foreach ($file in @($model.files)) {
            if (-not ([string]$file.path).ToLower().EndsWith("metadata.json")) { continue }
            $metaPath = Join-Path $dest ([string]$file.path)
            $meta = Get-Content -LiteralPath $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $onnxPath = ""
            foreach ($peer in @($model.files)) {
                if (([string]$peer.path).ToLower().EndsWith(".onnx")) {
                    $onnxPath = Join-Path $dest ([string]$peer.path)
                }
            }
            $meta | Add-Member -NotePropertyName "onnx_path" -NotePropertyValue $onnxPath -Force
            [System.IO.File]::WriteAllText($metaPath, ($meta | ConvertTo-Json -Depth 8), $utf8)
        }
    }
    Copy-Utf8 -Source (Join-Path $DemoModelsDir "README_示例模型.md") -Dest (Join-Path $dest "docs\示例模型说明.md")
    Write-Host "  已并入示例模型：$demoFiles 个文件（只有 ONNX，引擎仍需在目标机器上编）"
}

$stamp = @"
打包时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
编译解释器: $python
CUDA: 11.8
离线依赖包: wheels\（安装脚本用 pip --no-index 从这里装训练库，不联网）
说明: 本目录是给使用人员的安装包，不是源码仓库。
不含: 源码、示例模型、自动实验的 API Key 与 runtime 数据。
"@
[System.IO.File]::WriteAllText((Join-Path $dest "BUILD.txt"), $stamp, $utf8)

Write-Host "[7/7] 检查有没有把源码和密钥打进去"
$leaks = Get-ChildItem -LiteralPath $dest -Recurse -File | Where-Object {
    $_.Name -in @(
        "app.py", "config.py", "infer.py", "frozen_entry.py",
        "infer_worker.py", "inferproxy.py", "anomalydino_core.py",
        "tf_yolo_launcher_bin.py", "tf_anomalib_launcher_bin.py",
        "tf_anomalydino_launcher_bin.py"
    )
}
if ($leaks) {
    $leaks | ForEach-Object { Write-Host $_.FullName }
    throw "安装包里出现了源码文件，已停止。请不要把这个目录发出去。"
}
foreach ($stubName in @("tf_yolo_launcher.py", "tf_anomalib_launcher.py", "tf_anomalydino_launcher.py")) {
    $stubPath = Join-Path $dest "launchers\$stubName"
    $lines = @(Get-Content -LiteralPath $stubPath)
    if ($lines.Count -gt 12) { throw "$stubName 不是短启动桩，拒绝打包。" }
}
# 三个启动器必须都在：少一个，对应模型的「开始」按钮会报「找不到 launcher」
foreach ($pydName in @("tf_yolo_launcher_bin", "tf_anomalib_launcher_bin", "tf_anomalydino_launcher_bin")) {
    if (-not (Get-ChildItem -LiteralPath (Join-Path $dest "launchers") -Filter "$pydName*.pyd" -File)) {
        throw "launchers 目录缺少 $pydName.pyd"
    }
}
# 离线训练环境的两块拼图：wheels（库）+ 嵌入式 Python（解释器）。缺一块对方就装不出环境。
if (-not (Test-Path -LiteralPath (Join-Path $dest "wheels\manifest.txt"))) {
    throw "安装包缺少 wheels\manifest.txt，训练库没有带上。已停止打包。"
}
if (-not (Test-Path -LiteralPath (Join-Path $dest "runtime\python-embed.zip"))) {
    throw "安装包缺少 runtime\python-embed.zip，对方机器无法离线创建训练环境。已停止打包。"
}

# ---------------------------------------------------------------------------
# 密钥扫描：自动实验（LLM 决策层）的 API Key 绝不能进安装包
#
# Key 在开发机上有一份明文存在 6-web\runtime\autoexp_llm.json（网页上填过就落盘），
# 文件名看着像普通配置，且就在仓库目录里。本脚本目前不拷 runtime，
# 但"目前没拷"不是保证——这里显式拦一道，将来谁加了参数也发不出去。
# 细节与修法见 packaging\secret_scan.ps1。
# ---------------------------------------------------------------------------
Assert-NoPackagedSecrets -Root $dest -Label "软件安装包"

if (-not $SkipZip) {
    $zip = Join-Path $OutDir "TRTForgeWeb_cuda118.zip"
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    tar -a -c -f $zip -C $OutDir "TRTForgeWeb"
    if ($LASTEXITCODE -ne 0) { throw "压缩失败" }
    Write-Host "压缩包: $zip"
}

Write-Host ""
Write-Host "安装目录: $dest" -ForegroundColor Green
Write-Host "把整个 TRTForgeWeb 文件夹交给对方。对方双击「一键安装.bat」：第一次会装好训练环境、在桌面放图标并打开软件。以后双击桌面上的「TRTForge Web」。"
Write-Host ""
if (-not $WithDemoModels) {
    Write-Host "本包不含示例模型。需要演示数据时另打一个模型包（约 40 MB，只有 ONNX）：" -ForegroundColor Cyan
    Write-Host "    powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\package_demo_models.ps1" -ForegroundColor Cyan
    Write-Host "对方解压到软件目录旁边，双击 install_demo_models.bat 即可。" -ForegroundColor Cyan
}
