"""YOLO「用已有 best.pt 继续训练（微调）」的验证。

覆盖三层：
1. 命令行拼装：微调必须带 --init-mode finetune 与 --exist-ok，且不能和续训同时出现；
2. launcher：--exist-ok 透传成 Ultralytics 的 exist_ok，--init-mode 被接受；
3. 兼容性检查：任务头冲突要拦，类别差异要警告，写回目录要提前说明。

为什么要单独测：
微调写回同一个实验目录，一旦参数拼错或起点选错，覆盖的是用户几个小时的训练结果，
错误代价远高于「多跑一次」。这一层必须在开训前就把问题挡住。

用法：
    python backend/tools/test_finetune_request.py
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


def make_run(tmp: Path, task: str = "detect", names: list[str] | None = None, imgsz: int = 640) -> Path:
    """造一个最小可用的历史训练目录（与 Ultralytics 的落盘结构一致）。"""
    names = names or ["cat", "dog"]
    run_dir = tmp / "runs" / "exp_ft"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"fake-weights")
    (run_dir / "weights" / "last.pt").write_bytes(b"fake-weights")
    (run_dir / "args.yaml").write_text(
        "task: {task}\nmodel: yolov8n.pt\nimgsz: {imgsz}\ndata: {data}\nname: exp_ft\n"
        "optimizer: auto\nepochs: 10\n".format(
            task=task,
            imgsz=imgsz,
            data=str(tmp / "data.yaml").replace("\\", "/"),
        ),
        encoding="utf-8",
    )
    (run_dir / "results.csv").write_text(
        "epoch,time,metrics/mAP50-95(B)\n1,10,0.2\n2,20,0.3\n", encoding="utf-8"
    )
    return run_dir


def write_data_yaml(tmp: Path, names: list[str], filename: str = "data.yaml") -> Path:
    lines = ["path: .", "train: images/train", "val: images/val", "names:"]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    data = tmp / filename
    data.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def test_command_build(tmp: Path) -> None:
    print("\n[1] 命令行拼装")
    import app

    run_dir = make_run(tmp)
    data = write_data_yaml(tmp, ["cat", "dog"])
    best = str(run_dir / "weights" / "best.pt")

    base = dict(
        task="detect",
        model=best,
        data=str(data),
        epochs=30,
        imgsz=640,
        project=str(run_dir.parent),
        name=run_dir.name,
    )

    request = app.YoloTrainRequest(**base, init_mode="finetune")
    command = app.build_train_command(request, "python.exe")
    check("微调带 --init-mode finetune", "--init-mode" in command and "finetune" in command)
    check("微调带 --exist-ok（写回原目录）", "--exist-ok" in command)
    check("微调不带 --resume-from", "--resume-from" not in command)
    check("起点权重就是 best.pt", command[command.index("--model") + 1] == best)

    preset = app.YoloTrainRequest(**base, init_mode="preset")
    preset_cmd = app.build_train_command(preset, "python.exe")
    check("普通训练 --init-mode preset", preset_cmd[preset_cmd.index("--init-mode") + 1] == "preset")
    check("普通训练默认不覆盖目录", "--exist-ok" not in preset_cmd)

    # 学习率：微调没填时兜底 0.001（沿用 0.01 会冲掉原权重能力）；显式填了就用用户的
    check(
        "微调默认 lr0=0.001",
        "--lr0" in command and command[command.index("--lr0") + 1] == "0.001",
    )
    check("普通训练不传 lr0（用 Ultralytics 默认）", "--lr0" not in preset_cmd)
    tuned = app.YoloTrainRequest(**base, init_mode="finetune", lr0=0.0005)
    tuned_cmd = app.build_train_command(tuned, "python.exe")
    check(
        "微调显式 lr0 优先",
        tuned_cmd[tuned_cmd.index("--lr0") + 1] == "0.0005",
    )

    # 微调 + 续训互斥：否则 Ultralytics 以 resume 为准，表单参数全部失效
    mixed = app.YoloTrainRequest(**base, init_mode="finetune", resume_from=best)
    try:
        app.build_train_command(mixed, "python.exe")
        check("微调与续训互斥被拦住", False)
    except ValueError:
        check("微调与续训互斥被拦住", True)

    # 起点不存在必须报错，不能等到 Ultraalytics 启动才炸
    missing = app.YoloTrainRequest(**{**base, "model": str(tmp / "nope.pt")}, init_mode="finetune")
    try:
        app.build_train_command(missing, "python.exe")
        check("微调起点不存在被拦住", False)
    except ValueError:
        check("微调起点不存在被拦住", True)


def test_launcher_flags() -> None:
    print("\n[2] launcher 参数解析与 kwargs 透传")
    import tf_yolo_launcher as L

    parser = L.build_parser()
    args = parser.parse_args(
        ["--mode", "train", "--init-mode", "finetune", "--exist-ok", "--data", "x.yaml"]
    )
    check("launcher 接受 --init-mode finetune", args.init_mode == "finetune")
    check("launcher 接受 --exist-ok", args.exist_ok is True)

    default_args = parser.parse_args(["--mode", "train"])
    check("默认 init_mode=preset", default_args.init_mode == "preset")
    check("默认 exist_ok=False", default_args.exist_ok is False)

    tmp = Path(tempfile.mkdtemp(prefix="ftkw_"))
    data = tmp / "d.yaml"
    data.write_text("names: {0: ok}\n", encoding="utf-8")
    kwargs = L.build_train_kwargs(
        parser.parse_args(["--mode", "train", "--data", str(data), "--exist-ok"])
    )
    check("exist_ok 透传给 Ultralytics", kwargs.get("exist_ok") is True)
    kwargs_without = L.build_train_kwargs(parser.parse_args(["--mode", "train", "--data", str(data)]))
    check("未开启时不写入 exist_ok", "exist_ok" not in kwargs_without)


def test_compat_check(tmp: Path) -> None:
    print("\n[3] 微调兼容性检查")
    from checkpoints import finetune_check

    run_dir = make_run(tmp, names=["cat", "dog"])
    best = run_dir / "weights" / "best.pt"

    same = finetune_check(best, write_data_yaml(tmp, ["cat", "dog"]), "detect", 640)
    check("类别一致 -> 无类别警告", all("类别" not in item for item in same["warnings"]))
    check("提示写回原实验目录", any("写回原实验目录" in item for item in same["warnings"]))
    check("ok=True", same["ok"] is True)

    diff = finetune_check(best, write_data_yaml(tmp, ["cat", "truck"], "data_new.yaml"), "detect", 640)
    messages = " ".join(diff["warnings"])
    check("缺少旧类别被点名", "dog" in messages)
    check("新增类别被点名", "truck" in messages)

    mismatch = finetune_check(best, tmp / "data.yaml", "segment", 640)
    check("任务头不一致 -> blockers 非空", any("任务头不一致" in item for item in mismatch["blockers"]))
    check("任务头不一致 -> ok=False", mismatch["ok"] is False)

    resized = finetune_check(best, write_data_yaml(tmp, ["cat", "dog"]), "detect", 320)
    check("尺寸变化有提醒", any("输入尺寸" in item for item in resized["warnings"]))

    missing = finetune_check(tmp / "not-exist.pt", write_data_yaml(tmp, ["cat"]), "detect", 640)
    check("权重不存在 -> blocker", any("不存在" in item for item in missing["blockers"]))


def main() -> int:
    print("微调（用已有 best.pt 继续训练）验证")
    tmp = Path(tempfile.mkdtemp(prefix="finetune_"))
    test_command_build(tmp)
    test_launcher_flags()
    test_compat_check(tmp)

    print(f"\n结果：通过 {PASSED} 项，失败 {FAILED} 项")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
