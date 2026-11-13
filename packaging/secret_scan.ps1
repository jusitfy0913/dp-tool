# -*- coding: utf-8 -*-
# 安装包密钥扫描：打包脚本的最后一道关，被 package_web_user.ps1 与 package_demo_models.ps1 共用。
#
# 为什么必须有这道关
# ------------------
# 自动实验（LLM 决策层）的 API Key 在开发机上存在于两个地方：
#   1. 环境变量 AUTOEXP_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY；
#   2. 6-web/runtime/autoexp_llm.json —— 在网页上填过一次就会落盘。
#
# 第 2 条尤其危险：它在仓库目录里、文件名看着像普通配置。
# v1.35.3 起该文件里的 Key 已是 Windows DPAPI 密文，但**仍然必须拦**：
# 密文在原机器原账户上能解开，打包发出去等于顺手把钥匙寄给客户。
# 现在的打包脚本"碰巧没拷 runtime"，但这是**没有守卫的巧合**：
# 谁将来加一句 robocopy 6-web/runtime，Key 就跟着安装包发给客户，
# 发出去之后无法收回，只能换 Key。所以打包不能只依赖"没拷到"，
# 必须有一道显式检查在最后拦一下。
#
# 两道关，缺一不可
# ----------------
#   A. 文件名黑名单：专门存放密钥的文件一个都不许出现在包里；
#   B. 内容扫描：所有文本文件里出现密钥样式，或出现当前进程环境里的真实 Key，
#      就当打包失败。宁可让人改一次脚本，也不要把 Key 发出去。
#
# 用法（被 dot-source）：
#   . (Join-Path $PSScriptRoot "secret_scan.ps1")
#   Assert-NoPackagedSecrets -Root $dest -Label "软件包"

#: 环境变量里可能藏 Key 的名字。与 7-autoresearch-yolo\llm.py 的 KEY_ENVS 保持一致。
$script:SecretEnvNames = @(
    "AUTOEXP_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY"
)

#: 专门存密钥的文件名（小写比较）。这些文件出现即失败，不看内容。
$script:SecretFileNames = @(
    "autoexp_llm.json",
    "llm.local.json",
    "secrets.json",
    "credentials.json",
    "apikey.txt",
    "api_key.txt",
    ".env"
)

#: 二进制私钥容器：内容读不出来，只能按后缀一票否决。
$script:PrivateKeyContainers = @(".pfx", ".p12")

#: 文本型密钥文件后缀。**不能一律拒绝**：`app\certifi\cacert.pem` 是公开的 CA 根证书，
#: 打包时必然存在（certifi 依赖它），一律拒会让软件包永远打不出来。
#: 实测被误拦过一次，所以这里改成读内容判断：
#:   含 PRIVATE KEY        → 私钥，拒绝
#:   .key 且不含 CERTIFICATE → 来源不明的裸密钥，拒绝
#:   其余（含 BEGIN CERTIFICATE）→ 公开证书，放行
$script:PemLikeExtensions = @(".pem", ".key")

#: 要扫描内容的文本后缀。二进制（dll/exe/pyd/onnx/engine）不读，
#: 一方面是慢，另一方面是它们里面不会有可用的明文 Key。
$script:TextExtensions = @(
    ".json", ".txt", ".md", ".yaml", ".yml", ".js", ".mjs", ".html",
    ".css", ".py", ".ps1", ".bat", ".cmd", ".ini", ".cfg", ".log",
    ".csv", ".toml", ".xml", ".sh", ".conf", ".properties"
)

#: 内容模式。全部用**带前缀的字母数字**描述密钥形态，避免误伤普通单词。
$script:SecretPatterns = @(
    # DeepSeek / OpenAI 风格：sk- 后面跟 20 位以上
    @{ Name = "sk- 开头的密钥"; Pattern = 'sk-[A-Za-z0-9_\-]{20,}' },
    # JSON 里的 api_key 字段有实际值（空串不算）
    @{ Name = "api_key 字段有值"; Pattern = '"api[_-]?key"\s*:\s*"[^"\s]{8,}"' },
    # KEY=value / key: value 形态
    @{ Name = "KEY=value 形态"; Pattern = '(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key)\b\s*[:=]\s*["'']?[A-Za-z0-9_\-]{16,}' },
    # Authorization: Bearer xxx
    @{ Name = "Bearer 令牌"; Pattern = '(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}' },
    # Windows DPAPI 密文本体：Web 端把 Key 加密后存成这个字段。
    # 密文虽然解不开，但它**只对原机器原账户解开失败**——也就是说这仍是密钥材料，
    # 一旦跟着安装包发出去，用户会在自己的机器上看到一段来历不明的密文，
    # 而且原机器上还能解开。所以同样按密钥处理。
    @{ Name = "DPAPI 加密的 Key"; Pattern = '"api_key_dpapi"\s*:\s*"[^"\s]{16,}"' }
)

$script:SelfName = "secret_scan.ps1"


function Get-SecretEnvValues {
    <#
      取当前进程里真实存在的 Key 值。
      为什么要逐字比对：有些 Key 不带 sk- 前缀（自建网关、内部代理），
      只靠上面的模式会漏；但环境变量的值一定是我们最确知的那一份。
    #>
    $values = New-Object System.Collections.ArrayList
    foreach ($name in $script:SecretEnvNames) {
        $raw = [System.Environment]::GetEnvironmentVariable($name)
        if ($raw -and $raw.Trim().Length -ge 12) {
            [void]$values.Add(@{ Name = $name; Value = $raw.Trim() })
        }
    }
    return $values
}


function Format-SecretSnippet {
    <# 只留前 6 位，剩下的打码。日志里出现完整 Key 等于又泄露一次。 #>
    param([string]$Text)
    $text = "$Text"
    if ($text.Length -le 6) { return "***" }
    return $text.Substring(0, 6) + "***"
}


function Find-PackagedSecrets {
    <#
      扫描一个目录，返回发现列表（空列表表示干净）。
      每项：Path（相对路径）、Reason（中文原因）、Snippet（打码后的片段）。
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string[]]$ExtraSecrets = @()
    )

    $findings = New-Object System.Collections.ArrayList
    if (-not (Test-Path -LiteralPath $Root)) {
        throw "扫描目标不存在：$Root"
    }

    $envSecrets = Get-SecretEnvValues
    $literalSecrets = New-Object System.Collections.ArrayList
    foreach ($item in $envSecrets) { [void]$literalSecrets.Add($item) }
    foreach ($extra in $ExtraSecrets) {
        if ($extra -and $extra.Trim().Length -ge 12) {
            [void]$literalSecrets.Add(@{ Name = "调用方传入"; Value = $extra.Trim() })
        }
    }

    $rootFull = (Resolve-Path -LiteralPath $Root).Path

    foreach ($file in Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue) {
        $relative = $file.FullName.Substring($rootFull.Length).TrimStart("\", "/")
        $nameLower = $file.Name.ToLower()
        $extLower = $file.Extension.ToLower()

        # A. 文件名黑名单
        if ($nameLower -in $script:SecretFileNames) {
            [void]$findings.Add([pscustomobject]@{
                path = $relative
                reason = "文件名属于密钥类文件（这类文件就是专门存 Key 的）"
                snippet = ""
            })
            continue
        }
        if ($extLower -in $script:PrivateKeyContainers) {
            [void]$findings.Add([pscustomobject]@{
                path = $relative
                reason = "私钥容器文件（内容无法检查，按后缀否决）"
                snippet = ""
            })
            continue
        }
        if ($extLower -in $script:PemLikeExtensions) {
            # 只拒绝真私钥。`cacert.pem` 这种公开根证书必须放行，
            # 否则每个安装包都会在 certifi 上撞墙。
            $pemText = ""
            try {
                $pemText = [System.IO.File]::ReadAllText($file.FullName)
            } catch {
                $pemText = ""
            }
            if ($pemText.Contains("PRIVATE KEY")) {
                [void]$findings.Add([pscustomobject]@{
                    path = $relative
                    reason = "文件里有 PRIVATE KEY 段"
                    snippet = ""
                })
                continue
            }
            if ($extLower -eq ".key" -and -not $pemText.Contains("CERTIFICATE")) {
                [void]$findings.Add([pscustomobject]@{
                    path = $relative
                    reason = ".key 文件既不是证书也不像公开内容"
                    snippet = ""
                })
                continue
            }
            # 公开证书：不做内容模式扫描（一坨 base64 只会产生噪声）
            continue
        }
        # llm.local.json 之外的 *.local.json 也按同样的规矩处理：
        # 这套命名约定就是"本机私有配置"，不该进安装包。
        if ($nameLower.EndsWith(".local.json")) {
            [void]$findings.Add([pscustomobject]@{
                path = $relative
                reason = "文件名属于本机私有配置（*.local.json）"
                snippet = ""
            })
            continue
        }

        # B. 内容扫描（只读文本后缀；扫描器自己跳过，否则会匹配到自己的正则字面量）
        if ($nameLower -eq $script:SelfName) { continue }
        if ($extLower -notin $script:TextExtensions) { continue }

        $text = ""
        try {
            $text = [System.IO.File]::ReadAllText($file.FullName)
        } catch {
            # 读不了就跳过：编码怪异的文件不该让打包停摆，
            # 文件名那一道关对它依然有效。
            continue
        }
        if (-not $text) { continue }

        # 先做真实 Key 的逐字比对（最确凿，且能覆盖不带 sk- 前缀的 Key）
        foreach ($secret in $literalSecrets) {
            if ($text.Contains($secret.Value)) {
                [void]$findings.Add([pscustomobject]@{
                    path = $relative
                    reason = "文件里出现了环境变量 $($secret.Name) 的真实值"
                    snippet = (Format-SecretSnippet -Text $secret.Value)
                })
                continue
            }
        }

        foreach ($probe in $script:SecretPatterns) {
            $match = [regex]::Match($text, $probe.Pattern)
            if ($match.Success) {
                [void]$findings.Add([pscustomobject]@{
                    path = $relative
                    reason = "内容命中「$($probe.Name)」"
                    snippet = (Format-SecretSnippet -Text $match.Value)
                })
            }
        }
    }

    return $findings
}


function Assert-NoPackagedSecrets {
    <#
      打包脚本最后的统一关卡：有发现就抛异常，绝不"警告后继续"。
      刻意不支持"忽略"开关：这道关存在的意义就是不给例外。
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string]$Label = "安装包",
        [string[]]$ExtraSecrets = @()
    )

    Write-Host "  扫描密钥（文件名 + 文本内容）……"
    # 必须用 @(...) 收成数组：只命中一处时 PowerShell 会把单元素列表拆成标量，
    # 标量没有 .Count，后面那几句判断与提示都会变成空字符串，等于把失败静默掉。
    $findings = @(Find-PackagedSecrets -Root $Root -ExtraSecrets $ExtraSecrets)
    if ($findings.Count -eq 0) {
        Write-Host "  未发现密钥。" -ForegroundColor Green
        return
    }

    Write-Host ""
    Write-Host "$Label 里发现了 $($findings.Count) 处疑似密钥，已停止打包：" -ForegroundColor Red
    foreach ($item in $findings) {
        $extra = ""
        if ($item.snippet) { $extra = "　[$($item.snippet)]" }
        Write-Host ("    {0}{1}" -f $item.path, $extra) -ForegroundColor Red
        Write-Host ("      {0}" -f $item.reason) -ForegroundColor DarkYellow
    }
    Write-Host ""
    Write-Host "怎么修：" -ForegroundColor Yellow
    Write-Host "  1. 如果命中的是 6-web\runtime\autoexp_llm.json：把 runtime 目录从打包脚本里去掉。" -ForegroundColor Yellow
    Write-Host "     网页上填过的 Key 存在那个文件里，属于开发机本地数据，不该发给客户。" -ForegroundColor Yellow
    Write-Host "  2. 如果命中的是 *\configs\llm.local.json 或 *.local.json：那是本机私有配置，同理。" -ForegroundColor Yellow
    Write-Host "  3. 如果命中的是文档或示例：把 Key 换成占位符（例如 sk-xxxxxxxx）。" -ForegroundColor Yellow
    Write-Host "  4. Key 一旦真的随安装包发出去，请立刻去厂商后台吊销并重发，不要只删文件。" -ForegroundColor Yellow
    throw "$Label 未通过密钥扫描，已停止打包（共 $($findings.Count) 处）。"
}
