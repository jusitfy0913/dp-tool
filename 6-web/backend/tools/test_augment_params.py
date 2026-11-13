"""数据增强参数的端到端验证。

覆盖三层：
1. launcher 的 argparse 能解析所有增强参数，且 build_train_kwargs 把它们透传给 Ultralytics；
2. launcher 默认值必须与 Ultralytics DEFAULT_CFG 逐项一致（保证「不填 = 行为不变」）；
3. Web 后端只在用户填了值时才拼进命令行，并对越界值返回 400 而不是让训练跑到一半崩掉。

用法：
    python backend/tools/test_augment_params.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BACKEND_DIR.parent
LAUNCHER_DIR = WEB_DIR.parent / "1-python-yolo" / "launcher"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(LAUNCHER_DIR))

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


# 增强参数：launcher 的 --参数 名 -> 传入值 -> build_train_kwargs 里的键
AUG_CASES = [
    ("--hsv-h", "0.02", "hsv_h", 0.02),
    ("--hsv-s", "0.6", "hsv_s", 0.6),
    ("--hsv-v", "0.3", "hsv_v", 0.3),
    ("--degrees", "5", "degrees", 5.0),
    ("--translate", "0.2", "translate", 0.2),
    ("--scale", "0.4", "scale", 0.4),
    ("--shear", "2", "shear", 2.0),
    ("--perspective", "0.0005", "perspective", 0.0005),
    ("--flipud", "0.1", "flipud", 0.1),
    ("--fliplr", "0", "fliplr", 0.0),
    ("--bgr", "0.05", "bgr", 0.05),
    ("--mosaic", "0.1", "mosaic", 0.1),
    ("--mixup", "0.15", "mixup", 0.15),
    ("--copy-paste", "0.2", "copy_paste", 0.2),
    ("--close-mosaic", "20", "close_mosaic", 20),
]


def test_launcher() -> None:
    print("\n[1] launcher 参数解析与透传")
    import tf_yolo_launcher as L

    parser = L.build_parser()

    # 造一个最小可用的 dataset.yaml
    tmp = Path(tempfile.mkdtemp(prefix="augtest_"))
    data_yaml = tmp / "dataset.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames: {0: ok}\n", encoding="utf-8")

    # 1a. 每个参数单独传入都能透传到 kwargs
    for flag, raw, key, expected in AUG_CASES:
        argv = ["--mode", "train", "--data", str(data_yaml), flag, raw]
        args = parser.parse_args(argv)
        kwargs = L.build_train_kwargs(args)
        check(
            f"{flag} -> kwargs[{key}] == {expected}",
            kwargs.get(key) == expected,
            f"实际 {kwargs.get(key)!r}",
        )

    # 1b. 一次全传，确认互不干扰
    argv = ["--mode", "train", "--data", str(data_yaml)]
    for flag, raw, _key, _expected in AUG_CASES:
        argv += [flag, raw]
    kwargs = L.build_train_kwargs(parser.parse_args(argv))
    check("15 个参数同时传入全部生效", len([k for k in kwargs if k in {c[2] for c in AUG_CASES}]) == 15)

    # 1c. 默认值必须与 Ultralytics DEFAULT_CFG 完全一致
    print("\n[2] 默认值与 Ultralytics 对齐（关键：不填 = 行为不变）")
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT as ultralytics_defaults
    except Exception as exc:  # pragma: no cover - 环境缺 ultralytics 时跳过
        print(f"  [SKIP] 无法导入 ultralytics.cfg：{exc}")
    else:
        default_kwargs = L.build_train_kwargs(
            parser.parse_args(["--mode", "train", "--data", str(data_yaml)])
        )
        drift = {}
        for _flag, _raw, key, _expected in AUG_CASES:
            got = float(default_kwargs[key])
            want = float(ultralytics_defaults.get(key, 0))
            if abs(got - want) > 1e-12:
                drift[key] = (got, want)
        check("15 项默认值全部与 Ultralytics 一致", not drift, f"漂移项：{drift}")


def test_backend() -> None:
    print("\n[3] Web 后端命令拼装与越界校验")
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] 缺少 fastapi TestClient：{exc}")
        return

    import app as backend_app
    import web_state

    tmp = Path(tempfile.mkdtemp(prefix="augapi_"))
    data_yaml = tmp / "dataset.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames: {0: ok}\n", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    class FakeRun:
        def snapshot(self) -> dict:
            return {"run_id": "fake-run"}

    class FakeManager:
        def bind_loop(self, loop) -> None:  # lifespan 会调用，这里只做占位
            self.loop = loop

        def shutdown_all(self) -> None:  # lifespan 退出时调用
            pass

        def active_exclusive(self) -> list[dict]:
            # 并发闸门会问「有没有重任务在跑」；这里永远答「没有」，
            # 否则每个启动用例都会先撞上 409，测不到参数拼装
            return []

        def create(self, kind: str, command: list[str], cwd: str, **kwargs) -> FakeRun:
            # **kwargs：生产代码会带 warm_kind / meta，测试替身只关心 kind 与 command
            captured["kind"] = kind
            captured["command"] = command
            return FakeRun()

    # manager 是共享单例，放在 web_state 里；路由按 `web_state.manager` 动态取，
    # 所以只换这一处就够。不换的话第一发请求会真的起一个训练进程，第二发被 409 挡回来。
    original_manager = web_state.manager
    fake_manager = FakeManager()
    web_state.manager = fake_manager
    try:
        with TestClient(backend_app.app) as client:
            # 3a. 不填增强参数 -> 命令行里不应出现任何增强 flag
            captured.clear()
            resp = client.post("/api/train/yolo", json={"data": str(data_yaml)})
            check("不填增强参数时请求成功", resp.status_code == 200, str(resp.text))
            command = captured.get("command", [])
            aug_flags = [f for f, _r, _k, _e in AUG_CASES]
            leaked = [flag for flag in aug_flags if flag in command]
            check("不填时命令行为零增强 flag（行为不变）", not leaked, f"意外出现：{leaked}")

            # 3b. 填了就应原样拼进命令行
            captured.clear()
            resp = client.post(
                "/api/train/yolo",
                json={"data": str(data_yaml), "mosaic": 0.1, "close_mosaic": 20, "fliplr": 0.0},
            )
            check("填了增强参数时请求成功", resp.status_code == 200, str(resp.text))
            command = captured.get("command", [])
            check("--mosaic 0.1 已拼入", "--mosaic" in command and command[command.index("--mosaic") + 1] == "0.1")
            check("--close-mosaic 20 已拼入", "--close-mosaic" in command and command[command.index("--close-mosaic") + 1] == "20")
            # 0.0 是用户显式填的，不能被当成「没填」丢掉
            check("值为 0.0 时仍然拼入（不被误判为空）", "--fliplr" in command and command[command.index("--fliplr") + 1] == "0.0")

            # 3c. 越界值应返回 400 且不启动进程
            captured.clear()
            resp = client.post("/api/train/yolo", json={"data": str(data_yaml), "mosaic": 1.5})
            check("mosaic=1.5 返回 400", resp.status_code == 400, f"实际 {resp.status_code}")
            check("越界时未启动任何进程", "command" not in captured)

            captured.clear()
            resp = client.post("/api/train/yolo", json={"data": str(data_yaml), "close_mosaic": -1})
            check("close_mosaic=-1 返回 400", resp.status_code == 400, f"实际 {resp.status_code}")
    finally:
        web_state.manager = original_manager


def main() -> int:
    test_launcher()
    test_backend()
    print(f"\n{'=' * 56}")
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 56)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
