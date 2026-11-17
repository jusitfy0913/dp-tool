# -*- coding: utf-8 -*-
"""trtexec 额外参数白名单（v1.42.0）。

## 为什么必须白名单而不是黑名单

「追加参数」这个输入框会把内容原样拼到 trtexec 命令行尾部。trtexec 自己有
`--saveEngine` / `--exportOutput` / `--loadInputs` 这类**读写任意文件**的参数，
还有 `--plugins` / `--staticPlugins` 这类**加载外部 DLL** 的参数。也就是说，
不设防时它等价于「用服务账号执行任意文件读写 + 加载任意代码」。

黑名单永远漏（trtexec 每个版本都在加参数），所以只放行一份明确的、纯调优
性质的清单；不在清单里的直接拒绝，并在错误信息里把清单列出来——现场想加
参数时看得见该加什么，而不是收到一句「不允许」。

## 规则

1. 参数名按 `--name` 或 `--name=value` 匹配；
2. 值里出现 `/` 或 `\\` 一律拒绝（那意味着在读写在某个具体文件）；
3. 单个 `-` 开头的短参数不在清单内，同样拒绝。

清单来源：TensorRT 8.5 `trtexec --help` 中「只影响编译/测时行为、不碰文件」
的选项，逐条对着本项目的字段去重（`--fp16`、`--minShapes`、`--workspace`
这些已经由页面字段控制，不允许在这里重复给）。
"""

from __future__ import annotations

import shlex

#: 允许的参数名（原始大小写，用于错误信息与文档展示）。
ALLOWED_NAMES: tuple[str, ...] = (
    # 显存/工作区
    "memPoolSize",
    "builderOptimizationLevel",
    "maxAuxStreams",
    # tactic 选择
    "tacticSources",
    "noTF32",
    "sparsity",
    # 精度约束（只影响层，不引入新的校准流程）
    "precisionConstraints",
    "layerPrecisions",
    "inputIOFormats",
    "outputIOFormats",
    # 测时
    "avgTiming",
    "minTiming",
    "avgRuns",
    "iterations",
    "warmUp",
    "duration",
    "useCudaGraph",
    "noDataTransfers",
    "skipInference",
    # 日志
    "verbose",
    "profilingVerbosity",
)

#: 参数名比较用的小写集合（trtexec 自己区分大小写，但现场粘贴时常大小写混着写，
#: 这里放宽成不敏感；真正的安全边界是"在不在清单里"，不是大小写）。
ALLOWED_ARGS: frozenset[str] = frozenset(name.lower() for name in ALLOWED_NAMES)


def _arg_name(token: str) -> str:
    """从 ``--memPoolSize=workspace:4096`` 取出 ``mempoolsize``。"""
    name = token.lstrip("-")
    return name.split("=", 1)[0].strip().lower()


def _has_path(value: str) -> bool:
    """值里是否出现路径分隔符。出现即意味着指向某个具体文件。"""
    return "/" in value or "\\" in value


def split_args(raw: str) -> list[str]:
    """按 Windows 习惯切分参数串（保留引号内的空格）。

    ``posix=False`` 是有意的：``--layerPrecisions=conv1:fp16`` 这类值在 POSIX
    模式下会被进一步拆开，而 Windows 现场贴进来的路径引号也依赖非 POSIX 规则。
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return shlex.split(text, posix=False)


def check(raw: str) -> list[str]:
    """校验并返回参数列表；有任何一个不在白名单就抛 ``ValueError``。

    调用方（路由）把 ``ValueError`` 转成 400，错误信息里带上允许清单。
    """
    tokens = split_args(raw)
    if not tokens:
        return []

    rejected: list[str] = []

    def reject(item: str) -> None:
        if item not in rejected:
            rejected.append(item)

    for token in tokens:
        if token.startswith("-"):
            if _arg_name(token) not in ALLOWED_ARGS:
                reject(token)
            # 内联值（--a=b）也要检查路径
            if "=" in token and _has_path(token.split("=", 1)[1]):
                reject(token)
            continue
        # 独立的值（--iterations 100 这种写法）
        if _has_path(token):
            reject(token)

    if rejected:
        raise ValueError(
            "额外参数里包含不允许的项："
            + "、".join(rejected)
            + f"。只放行这些调优参数：{describe_allowed()}。"
            "读写文件、加载插件、选择精度这类参数请用页面上对应的字段。"
        )
    return tokens


def describe_allowed() -> str:
    """给前端/文档用的一行说明。"""
    return "、".join(f"--{name}" for name in sorted(ALLOWED_NAMES, key=str.lower))
