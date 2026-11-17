# -*- coding: utf-8 -*-
"""训练命令行快照测试：新增 AnomalyDINO 不能改动三个旧模型拼出来的参数。

守的是一类很隐蔽的回归：为了把零样本模型塞进同一个请求模型，顺手把
`--max-epochs` 的类型换了、或者把某个参数挪到了 `if` 里。前端表单没变、
接口也没变，只有训练产物悄悄不同（比如阈值校准目录被丢掉）。

所以这里逐字符比对命令行：旧模型必须与 v1.39.0 完全一致，
新模型的参数必须都能被启动器的 argparse 认出来。

    python 6-web/backend/tools/test_anomalib_commands.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from _harness import check, finish  # noqa: E402

import config  # noqa: E402
from routers import train  # noqa: E402


def normalize(command: list[str]) -> list[str]:
    """把随机器变化的两个绝对路径换成占位符，其余原样比较。"""
    out: list[str] = []
    for item in command:
        text = str(item)
        if text == "PY":
            out.append("PY")
        elif Path(text).name == "tf_anomalib_launcher.py":
            out.append("<ANOMALIB>")
        elif Path(text).name == "tf_anomalydino_launcher.py":
            out.append("<ANOMALYDINO>")
        elif text.startswith(str(Path.home())):
            out.append("<HOME>" + text[len(str(Path.home())):])
        else:
            out.append(text)
    return out


BASE_ARGS = [
    "--root", "data\\bottle",
    "--normal-dir", "train/good",
    "--abnormal-dir", "test/abnormal",
    "--normal-test-dir", "test/good",
    "--project", "results",
    "--name", "latest",
    "--input-size", "256",
    "--max-epochs", "auto",
    "--train-batch-size", "4",
    "--eval-batch-size", "4",
    "--num-workers", "0",
    "--val-split-ratio", "0.2",
    "--calibrate-metadata", "true",
    "--calib-limit", "64",
]


def make_request(model: str, **overrides) -> train.AnomalibTrainRequest:
    payload = {"model": model, "root": "data/bottle"}
    payload.update(overrides)
    return train.AnomalibTrainRequest(**payload)


def launcher_flags(source: Path) -> set[str]:
    text = source.read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*"(--[\w-]+)', text))


def main() -> int:
    try:
        # ---- 1. 三个旧模型的命令行逐字符不变 -----------------------------
        for model in ("patchcore", "padim", "efficientad"):
            expected = ["PY", "<ANOMALIB>", "--mode", "train-export", "--model", model]
            expected += BASE_ARGS
            actual = normalize(train.build_anomalib_train_command(make_request(model), "PY"))
            check(
                f"{model} 命令行未被改动",
                actual == expected,
                f"\n  期望 {expected}\n  实际 {actual}",
            )

        # ---- 2. 可选目录只在非空时才拼 ------------------------------------
        request = make_request(
            "patchcore",
            imagenet_dir="  data/imagenette  ",
            calib_normal_dir="data/calib/good",
            calib_abnormal_dir="",
        )
        command = normalize(train.build_anomalib_train_command(request, "PY"))
        check("imagenet 目录去空格后传入",
              command[command.index("--imagenet-dir") + 1] == "data/imagenette", str(command))
        check("校准目录非空则传入", "--calib-normal-dir" in command)
        check("空校准目录不传", "--calib-abnormal-dir" not in command)

        # ---- 3. AnomalyDINO 走另一个启动器与模式 ---------------------------
        dino = normalize(train.build_anomalydino_command(make_request("anomalydino"), "PY"))
        check("用 AnomalyDINO 启动器", dino[1] == "<ANOMALYDINO>", str(dino[:2]))
        check("模式是 build-export",
              dino[dino.index("--mode") + 1] == "build-export", str(dino[:4]))
        check("不再传 --model（启动器无第二个模型）", "--model" not in dino)

        # ---- 4. 零样本专属参数原样落到命令行 -------------------------------
        dino = normalize(
            train.build_anomalydino_command(
                make_request("anomalydino", shots=24, rotation=True, seed=3), "PY"
            )
        )
        for flag, want in (("--shots", "24"), ("--rotation", "true"), ("--seed", "3")):
            ok = flag in dino and dino[dino.index(flag) + 1] == want
            check(f"{flag} 传 {want}", ok, str(dino))
        default_dino = normalize(
            train.build_anomalydino_command(make_request("anomalydino"), "PY")
        )
        check("rotation 默认关闭时传 false",
              default_dino[default_dino.index("--rotation") + 1] == "false", str(default_dino))

        # ---- 5. 命令行里的每个参数启动器都得认识 ---------------------------
        flags = launcher_flags(Path(config.ANOMALYDINO_LAUNCHER))
        check("能读到启动器的参数表", bool(flags), str(len(flags)))
        unknown = sorted(flag for flag in dino if flag.startswith("--") and flag not in flags)
        check("命令行参数都被启动器支持", not unknown, "多出的参数：" + ", ".join(unknown))

        # ---- 6. 兼容旧字段：launcher 也认 Anomalib 那批参数名 ---------------
        for flag in ("--mode", "--root", "--normal-dir", "--project", "--name",
                     "--input-size", "--max-epochs", "--calib-limit"):
            check(f"启动器接受 {flag}", flag in flags)

        # ---- 7. 分派函数给出正确的工作目录 ---------------------------------
        command, cwd = train.build_train_export_command(make_request("anomalydino"), "PY")
        check("分派到 AnomalyDINO 启动器", Path(command[1]).name == "tf_anomalydino_launcher.py",
              str(command[1]))
        check("工作目录是启动器目录", Path(cwd) == Path(config.ANOMALIB_LAUNCHER_DIR),
              f"{cwd} != {config.ANOMALIB_LAUNCHER_DIR}")

        command, cwd = train.build_train_export_command(make_request("patchcore"), "PY")
        check("旧模型仍分派到 Anomalib 启动器",
              Path(command[1]).name == "tf_anomalib_launcher.py", str(command[1]))

        # ---- 8. 全流程请求模型接受 anomalydino ------------------------------
        from routers import pipeline as pipeline_router

        spec = pipeline_router.AnomalibPipelineRequest(model="anomalydino", root="data/bottle")
        check("全流程请求接受 anomalydino", spec.model == "anomalydino")
        spec = pipeline_router.AnomalibPipelineRequest(model="anomalydino", root="x", shots=32)
        check("全流程请求透传 shots", spec.shots == 32)
        spec = pipeline_router.AnomalibPipelineRequest(model="patchcore", root="x")
        check("全流程请求默认仍是 patchcore", spec.model == "patchcore")
    finally:
        pass

    return finish()


if __name__ == "__main__":
    sys.exit(main())
