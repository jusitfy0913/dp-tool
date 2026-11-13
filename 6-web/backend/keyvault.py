# -*- coding: utf-8 -*-
"""API Key 的本机加密存储（Windows DPAPI）。

为什么需要它
------------
自动实验（LLM 决策层）的 Key 原来以**明文**存在 ``6-web/runtime/autoexp_llm.json``。
那个文件在仓库目录里、名字像普通配置，只要有人拷走整个文件夹（U 盘、网盘同步、
打包脚本加错一句 robocopy），Key 就跟着走了，而且事后无法追回——只能去厂商后台吊销。

加密方案的选择
--------------
用 Windows 自带的 **DPAPI**（``CryptProtectData`` / ``CryptUnprotectData``），不自己造密码学，
也不引入 pywin32 这类重依赖——直接 ``ctypes`` 调 crypt32：

| 方案 | 为什么不用 / 为什么用 |
| --- | --- |
| 自己写 AES + 固定密钥 | 密钥还得存在某处，等于把锁和钥匙放同一个抽屉 |
| 密码派生（PBKDF2） | 每次启动都要用户输密码，网页服务没法无人值守启动 |
| 机器绑定的加密 | 换台机器拷贝走密文反而**能**解开，正好是我们要防的场景之一 |
| **DPAPI（默认用户作用域）** | 密钥由 Windows 用「当前用户登录凭据」派生并托管；密文离开这台机器或换个账户，**解不开**。这正是需求 |

代价（必须写清楚，不能假装没有）
--------------------------------
密文**绑定当前 Windows 用户 + 这台机器**。换了电脑、重装系统、换了 Windows 账户，
旧密文一律解不开。这不是缺陷，是设计目标：那种情况下用户重新输一次 Key。
代码里对应 ``_key_locked`` 状态，界面会明说「请重新输入」，而不是显示「未配置」让人以为丢数据了。

降级策略
--------
非 Windows 平台没有 DPAPI，退回明文存储并在文件里标 ``key_enc = "plain"``。
**Windows 上加密失败则直接报错，绝不静默写明文**——否则这道防线会在最需要它的时候自己消失。
"""

from __future__ import annotations

import base64
import os
import sys

#: 密文里绑定的附加熵。它不是密钥（DPAPI 的真实密钥由系统托管），
#: 作用是让「用本模块加密的 blob」无法被同一用户下别的程序解出的 blob 顶替。
#: ⚠ 改这个字符串等于让所有已保存的 Key 立即失效（会落到「请重新输入」），
#: 所以只在存储格式真的变了的时候才动，并且要同步升级 vN 后缀。
ENTROPY = b"trtforge.autoexp.llm.v1"

#: 告诉系统不要弹任何 UI（服务进程没有桌面，弹窗会直接卡死）。
_CRYPTPROTECT_UI_FORBIDDEN = 0x01

#: 存储文件里标记加密方式的字段名。读的时候先看它，不靠猜。
FIELD_CIPHERTEXT = "api_key_dpapi"
FIELD_SCHEME = "key_enc"

SCHEME_DPAPI = "dpapi"
SCHEME_PLAIN = "plain"


class KeyVaultError(RuntimeError):
    """加解密失败。``reason`` 是给人看的中文说明。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def is_windows() -> bool:
    return os.name == "nt" and sys.platform.startswith("win")


def available() -> bool:
    """本机是否具备 DPAPI。只判断平台，不实际调用（调用有副作用风险且更慢）。"""
    return is_windows()


# ---------------------------------------------------------------------------
# DPAPI 绑定（惰性导入并缓存函数指针）
#
# 为什么惰性：非 Windows 机器 import 本模块不能失败，
# 否则后端在 Linux 上直接起不来——而 Linux 只是不加密，不是不能用。
# ---------------------------------------------------------------------------
_CRYPT32 = None
_KERNEL32 = None
_BLOB = None


def _bind() -> tuple:
    global _CRYPT32, _KERNEL32, _BLOB
    if _CRYPT32 is not None:
        return _CRYPT32, _KERNEL32, _BLOB

    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # 显式声明参数类型：不声明的话 64 位下指针会被当成 32 位 int，
    # 表现为「偶尔成功、偶尔 0xC0000005 崩在 crypt32 里」，极难查。
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DATA_BLOB), wintypes.LPCWSTR, ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DATA_BLOB),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DATA_BLOB), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DATA_BLOB),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    _CRYPT32, _KERNEL32, _BLOB = crypt32, kernel32, DATA_BLOB
    return _CRYPT32, _KERNEL32, _BLOB


def _make_blob(data: bytes):
    """把 bytes 包成 DATA_BLOB，同时返回源缓冲区的引用。

    缓冲区必须由调用方持住：只返回结构体的话，create_string_buffer 出来的对象
    会在函数返回时被 GC，DPAPI 读到的就是已释放内存（随机崩溃或数据错乱）。
    """
    import ctypes

    blob_type = _BLOB
    buffer = ctypes.create_string_buffer(data, max(1, len(data)))
    blob = blob_type(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    return blob, buffer


def protect(text: str) -> str:
    """加密并返回 base64 密文。失败抛 :class:`KeyVaultError`。

    Windows 上失败**不会**退化成明文；调用方要么把错误报给用户，要么不写文件。
    """
    if not is_windows():
        raise KeyVaultError("当前平台没有 DPAPI（只有 Windows 提供）")
    if not text:
        raise KeyVaultError("空字符串没有加密的意义")

    crypt32, _kernel32, _ = _bind()
    secret_buf = _make_blob(text.encode("utf-8"))
    entropy_buf = _make_blob(ENTROPY)
    out = _BLOB()

    ok = crypt32.CryptProtectData(
        _byref(secret_buf[0]), None, _byref(entropy_buf[0]),
        None, None, _CRYPTPROTECT_UI_FORBIDDEN, _byref(out),
    )
    if not ok:
        raise KeyVaultError(f"DPAPI 加密失败：{_last_error()}")
    try:
        cipher = _read_blob(out)
    finally:
        _local_free(out)

    return base64.b64encode(cipher).decode("ascii")


def unprotect(blob_b64: str) -> str:
    """解密 base64 密文。解不开时抛 :class:`KeyVaultError`。

    这是「换电脑 / 换账户 / 文件被改」三种情况的统一出口，
    调用方据此把状态标成「需要重新输入 Key」，而不是当成「没配 Key」。
    """
    if not is_windows():
        raise KeyVaultError("当前平台没有 DPAPI（只有 Windows 提供）")
    text = (blob_b64 or "").strip()
    if not text:
        raise KeyVaultError("密文为空")
    try:
        cipher = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as exc:
        raise KeyVaultError(f"密文不是合法 base64：{exc}") from exc

    crypt32, _kernel32, _ = _bind()
    secret_buf = _make_blob(cipher)
    entropy_buf = _make_blob(ENTROPY)
    out = _BLOB()

    ok = crypt32.CryptUnprotectData(
        _byref(secret_buf[0]), None, _byref(entropy_buf[0]),
        None, None, _CRYPTPROTECT_UI_FORBIDDEN, _byref(out),
    )
    if not ok:
        # 这里刻意不把系统错误码直接抛给用户：常见原因是「不是这台机器/这个账户」，
        # 原始错误码（-2146893813 之类）对用户没有任何帮助。
        raise KeyVaultError("解不开已保存的 Key（通常是换了电脑或 Windows 账户，请重新输入）")
    try:
        raw = _read_blob(out)
    finally:
        _local_free(out)

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KeyVaultError(f"解密结果不是合法 UTF-8：{exc}") from exc


# ---------------------------------------------------------------------------
# ctypes 小工具。写出来是为了让上面的主流程可读——
# 每个函数只做一件事，且都在失败时给中文原因而不是裸 OSError。
# ---------------------------------------------------------------------------
def _byref(blob):
    import ctypes

    return ctypes.byref(blob)


def _read_blob(blob) -> bytes:
    import ctypes

    if not blob.pbData or blob.cbData == 0:
        return b""
    return ctypes.string_at(blob.pbData, blob.cbData)


def _local_free(blob) -> None:
    """释放 DPAPI 分配的缓冲区。不释放就是每次保存/读取泄漏一小块进程内存。

    注意必须传**指针本身**（``pbData`` 的值），不是 ``byref(pbData)``——
    后者给出去的是「指针变量所在地址」，LocalFree 会去释放一块不属于堆的内存，
    表现为进程直接 0xC0000374（堆损坏）崩溃，而且崩在 crypt32 内部，极难定位。
    """
    import ctypes

    if not blob.pbData:
        return
    try:
        _KERNEL32.LocalFree(ctypes.cast(blob.pbData, ctypes.c_void_p))
    except Exception:  # noqa: BLE001 - 释放失败不该影响功能
        pass


def _last_error() -> str:
    import ctypes

    code = ctypes.get_last_error()
    return f"错误码 {code}" if code else "未知原因"


def describe() -> dict:
    """给接口和界面用的状态摘要（不含任何密钥材料）。"""
    return {
        "available": available(),
        "scheme": SCHEME_DPAPI if available() else SCHEME_PLAIN,
        "scope": "current_user" if available() else "none",
        "note": (
            "Key 用 Windows DPAPI 加密，只有这台电脑的这个 Windows 账户能解开；"
            "换电脑或换账户需要重新输入"
        ) if available() else (
            "当前平台没有 DPAPI，Key 以明文保存（仅限开发环境）"
        ),
    }
