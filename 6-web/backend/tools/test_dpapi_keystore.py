# -*- coding: utf-8 -*-
"""API Key 的 DPAPI 加密存储测试（keyvault + autoexp_api 的读写层）。

    python tools/test_dpapi_keystore.py

测什么
------
1. **磁盘上没有明文**。这是这个功能存在的全部理由：文件被拷走也读不出 Key。
2. **往返正确**。加密再解开必须一模一样，否则用户下次开跑时 Key 是错的。
3. **换机/换账户的模拟**（伪造一段别的机器产生的密文）：不许崩、不许当成
   「没配 Key」，必须落到 ``key_locked`` 状态并给出「请重新输入」。
4. **旧格式明文自动迁移**。升级前存过 Key 的用户不该需要手动改文件。
5. **加密失败绝不写明文**。Windows 上 DPAPI 出错时保存要失败，而不是悄悄降级。

所有用例都把 ``LLM_STORE`` 指到临时目录，绝不碰真实的
``6-web/runtime/autoexp_llm.json``。
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import keyvault  # noqa: E402
import autoexp_api  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []

SECRET = "sk-dpapi-test-9f3a7c1e5b2d"


def case(name: str):
    def wrap(func):
        def runner():
            try:
                func()
            except Exception as exc:  # noqa: BLE001
                FAILED.append((name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
                print(f"  [FAIL] {name}")
            else:
                PASSED.append(name)
                print(f"  [ ok ] {name}")

        runner.__name__ = func.__name__
        return runner

    return wrap


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}：期望 {expected!r}，实际 {actual!r}")


class TempStore:
    """把 LLM_STORE 临时指到别处，并保证退出时恢复原值。"""

    def __enter__(self):
        self.original = autoexp_api.LLM_STORE
        self._tmp = tempfile.TemporaryDirectory()
        autoexp_api.LLM_STORE = Path(self._tmp.name) / "autoexp_llm.json"
        return autoexp_api.LLM_STORE

    def __exit__(self, *exc_info):
        autoexp_api.LLM_STORE = self.original
        self._tmp.cleanup()
        return False


class ForceVault:
    """临时改写「本机是否有 DPAPI / 加密是否成功」，用来测降级路径。

    直接 monkeypatch ``keyvault`` 上的函数，而不是复制一套代码——
    测的必须是被测代码本身的行为，而不是测试自己写的行为。
    """

    def __init__(self, *, available: bool | None = None, protect_fails: bool = False):
        self._available = available
        self._protect_fails = protect_fails

    def __enter__(self):
        self._orig_available = keyvault.available
        self._orig_protect = keyvault.protect
        if self._available is not None:
            keyvault.available = lambda: self._available
        if self._protect_fails:
            def boom(_text):
                raise keyvault.KeyVaultError("模拟 DPAPI 故障")

            keyvault.protect = boom
        return self

    def __exit__(self, *exc):
        keyvault.available = self._orig_available
        keyvault.protect = self._orig_protect
        return False


# ---------------------------------------------------------------------------
# keyvault 本体
# ---------------------------------------------------------------------------

@case("keyvault：加密后的密文里不出现明文 Key")
def test_cipher_has_no_plaintext():
    if not keyvault.available():
        print("        （跳过：本机没有 DPAPI）")
        return
    cipher = keyvault.protect(SECRET)
    check(SECRET not in cipher, "base64 密文里出现了明文")
    raw = base64.b64decode(cipher)
    check(SECRET.encode() not in raw, "二进制密文里出现了明文")


@case("keyvault：加解密往返一致")
def test_roundtrip():
    if not keyvault.available():
        print("        （跳过：本机没有 DPAPI）")
        return
    for text in ("sk-a", "sk-" + "x" * 200, "带中文的密钥", SECRET):
        check_eq(keyvault.unprotect(keyvault.protect(text)), text, f"往返 {text[:8]}")


@case("keyvault：两次加密同一明文得到不同密文（DPAPI 自带随机盐）")
def test_not_deterministic():
    if not keyvault.available():
        print("        （跳过：本机没有 DPAPI）")
        return
    check(keyvault.protect(SECRET) != keyvault.protect(SECRET), "两次密文相同")


@case("keyvault：非法输入一律抛 KeyVaultError，不抛裸异常")
def test_bad_inputs():
    if not keyvault.available():
        print("        （跳过：本机没有 DPAPI）")
        return
    for bad in ("", "not base64 !!", base64.b64encode(b"random bytes").decode()):
        try:
            keyvault.unprotect(bad)
        except keyvault.KeyVaultError:
            continue
        raise AssertionError(f"应拒绝非法密文：{bad[:16]!r}")


@case("keyvault：别的机器/账户产生的密文解不开（用另一份 blob 模拟）")
def test_foreign_blob():
    if not keyvault.available():
        print("        （跳过：本机没有 DPAPI）")
        return
    # 真实的跨机密文没法在本机造出来，这里用「合法 base64 + 非本模块密文」模拟：
    # 系统同样会拒绝，走的是与换机完全相同的失败分支。
    foreign = base64.b64encode(b"\x01\x02\x03\x04" * 40).decode()
    try:
        keyvault.unprotect(foreign)
    except keyvault.KeyVaultError as exc:
        check("重新输入" in exc.reason, f"提示应引导用户重输，实际：{exc.reason}")
        return
    raise AssertionError("外来密文不应被解开")


@case("keyvault：describe 不含任何密钥材料")
def test_describe_has_no_secret():
    info = keyvault.describe()
    check("available" in info, "缺少 available")
    check("scheme" in info, "缺少 scheme")
    text = json.dumps(info, ensure_ascii=False)
    check(SECRET not in text, "describe 泄漏了明文 Key")
    check("sk-" not in text, "describe 出现了疑似 Key 的片段")


# ---------------------------------------------------------------------------
# autoexp_api 的读写层
# ---------------------------------------------------------------------------

@case("保存后磁盘文件里没有明文 Key，只有 dpapi 密文")
def test_disk_has_no_plaintext():
    with TempStore() as store:
        autoexp_api._write_store({"api_key": SECRET, "model": "deepseek-chat"})
        raw = store.read_text(encoding="utf-8")
        check(SECRET not in raw, f"磁盘上出现了明文 Key：{raw[:200]}")
        data = json.loads(raw)
        if keyvault.available():
            check("api_key" not in data, "Windows 上不应再写明文 api_key 字段")
            check(bool(data.get(keyvault.FIELD_CIPHERTEXT)), "缺少 api_key_dpapi 字段")
            check_eq(data.get(keyvault.FIELD_SCHEME), "dpapi", "加密方式标记")
        else:
            check_eq(data.get(keyvault.FIELD_SCHEME), "plain", "非 Windows 应标 plain")


@case("读取时自动解密：业务层拿到的就是明文 Key")
def test_read_decrypts():
    with TempStore():
        autoexp_api._write_store({"api_key": SECRET})
        store = autoexp_api._read_store()
        check_eq(store.get("api_key"), SECRET, "读出的 Key")
        check(not store.get("_key_locked"), "不应被标记为解不开")
        check(keyvault.FIELD_CIPHERTEXT not in store, "业务视图不该暴露密文字段")


@case("旧格式明文文件能读出来（向后兼容）")
def test_legacy_plaintext_read():
    with TempStore() as store:
        store.write_text(json.dumps({"api_key": SECRET, "model": "m"}),
                         encoding="utf-8")
        check_eq(autoexp_api._read_store().get("api_key"), SECRET, "旧格式明文")


@case("启动迁移：旧明文就地变成密文，且值不变")
def test_migration():
    with TempStore() as store:
        store.write_text(json.dumps({"api_key": SECRET, "base_url": "https://x"}),
                         encoding="utf-8")
        result = autoexp_api.migrate_key_store()
        if not keyvault.available():
            print("        （跳过断言：本机没有 DPAPI）")
            return
        check(result["migrated"] is True, f"应完成迁移，实际：{result}")
        raw = store.read_text(encoding="utf-8")
        check(SECRET not in raw, "迁移后磁盘仍有明文")
        check_eq(autoexp_api._read_store().get("api_key"), SECRET, "迁移后仍能读出原值")
        check_eq(autoexp_api._read_store().get("base_url"), "https://x", "其它字段不丢")


@case("迁移是幂等的：已经是密文时不做任何事")
def test_migration_idempotent():
    with TempStore():
        autoexp_api._write_store({"api_key": SECRET})
        first = autoexp_api.migrate_key_store()
        second = autoexp_api.migrate_key_store()
        check(first["migrated"] is False, f"第二次不该再迁移：{first}")
        check(second["migrated"] is False, f"第三次不该再迁移：{second}")


@case("换机场景：密文解不开时标记 key_locked，而不是当作没配")
def test_locked_state():
    with TempStore() as store:
        foreign = base64.b64encode(b"\x09" * 64).decode()
        store.write_text(json.dumps({"api_key_dpapi": foreign, "key_enc": "dpapi"}),
                         encoding="utf-8")
        state = autoexp_api._llm_state()
        if not keyvault.available():
            print("        （跳过：本机没有 DPAPI）")
            return
        check_eq(state["has_key"], False, "解不开时 has_key")
        check_eq(state["key_locked"], True, "解不开时应标 key_locked")
        check_eq(state["key_masked"], "", "解不开时不该有遮罩")


@case("换机场景：只改 base_url 不写 Key，旧密文必须保留")
def test_locked_preserved_on_other_save():
    with TempStore() as store:
        foreign = base64.b64encode(b"\x07" * 64).decode()
        store.write_text(json.dumps({"api_key_dpapi": foreign}), encoding="utf-8")
        if not keyvault.available():
            print("        （跳过：本机没有 DPAPI）")
            return
        view = autoexp_api._read_store()
        view["base_url"] = "https://api.deepseek.com"
        autoexp_api._write_store(view)
        data = json.loads(store.read_text(encoding="utf-8"))
        check_eq(data.get("api_key_dpapi"), foreign,
                 "未提供新 Key 时旧密文应原样保留（拷回原机器还能用）")


@case("重新输入 Key 后清除 key_locked")
def test_rewrite_clears_locked():
    with TempStore() as store:
        foreign = base64.b64encode(b"\x05" * 64).decode()
        store.write_text(json.dumps({"api_key_dpapi": foreign}), encoding="utf-8")
        if not keyvault.available():
            print("        （跳过：本机没有 DPAPI）")
            return
        check_eq(autoexp_api._read_store().get("_key_locked"), True, "初始应为 locked")
        autoexp_api._write_store({"api_key": "sk-new-key-1234"})
        check_eq(autoexp_api._read_store().get("api_key"), "sk-new-key-1234", "新 Key")
        check(not autoexp_api._read_store().get("_key_locked"), "应清掉 locked 标记")


@case("清空 Key：两个字段一起删，不留残骸")
def test_clear_removes_both():
    with TempStore() as store:
        autoexp_api._write_store({"api_key": SECRET})
        autoexp_api._write_store({"model": "deepseek-chat"})
        data = json.loads(store.read_text(encoding="utf-8"))
        check("api_key" not in data, "明文 api_key 应被删除")
        check("api_key_dpapi" not in data, "密文字段应被删除")
        check_eq(data.get("model"), "deepseek-chat", "其它字段应保留")


@case("非 Windows 降级：标 plain 并且仍能读回")
def test_plain_fallback():
    with TempStore() as store, ForceVault(available=False):
        autoexp_api._write_store({"api_key": SECRET})
        data = json.loads(store.read_text(encoding="utf-8"))
        check_eq(data.get("key_enc"), "plain", "降级标记")
        check_eq(data.get("api_key"), SECRET, "降级时应写明文（否则读完就丢了）")
        check_eq(autoexp_api._read_store().get("api_key"), SECRET, "降级后可读")


@case("Windows 上加密失败：保存必须失败，绝不写明文")
def test_encrypt_failure_never_plaintext():
    with TempStore() as store, ForceVault(available=True, protect_fails=True):
        raised = None
        try:
            autoexp_api._write_store({"api_key": SECRET})
        except OSError as exc:
            raised = exc
        check(raised is not None, "加密失败时应抛 OSError 让接口返回 500")
        check("明文" in str(raised), f"错误信息应说明避免了明文落盘：{raised}")
        check(not store.exists(), "加密失败时不应留下任何文件")


@case("配置文件损坏：读成空配置，不抛异常")
def test_broken_json():
    with TempStore() as store:
        store.write_text("{not json", encoding="utf-8")
        check_eq(autoexp_api._read_store(), {}, "损坏文件应读成空")
        check_eq(autoexp_api.migrate_key_store()["migrated"], False, "损坏文件不迁移")


@case("启动钩子确实调用了迁移（静态检查）")
def test_startup_wired():
    app_py = (BACKEND / "app.py").read_text(encoding="utf-8")
    check("migrate_key_store" in app_py, "app.py 的 lifespan 里应调用 autoexp_api.migrate_key_store")


@case("Key 在 DPAPI 密文里不会随 AUTOEXP_LLM_CONFIG 漏进子进程配置")
def test_llm_config_strips_cipher():
    llm_py = (BACKEND.parent.parent / "7-autoresearch-yolo" / "llm.py")
    text = llm_py.read_text(encoding="utf-8")
    check("api_key_dpapi" in text, "llm.load_config 应剔除 api_key_dpapi，避免密文流进台账")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"API Key DPAPI 存储测试（{BACKEND}）")
    print(f"Python {sys.version.split()[0]}  os={os.name}  dpapi={keyvault.available()}")
    print()

    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()

    print()
    print("=" * 60)
    total = len(PASSED) + len(FAILED)
    print(f"通过 {len(PASSED)} / {total}")
    if FAILED:
        print()
        for name, detail in FAILED:
            print(f"[FAIL] {name}")
            print(detail)
            print("-" * 60)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
