# -*- coding: utf-8 -*-
"""训练默认值护栏：Web 请求模型的默认值必须与启动器 argparse 默认值一致。

## 为什么需要这个文件

「Web 上不填参数直接点开始」和「命令行直接跑启动器」应该得到同一个训练。
这条在 v1.41 之前就已经破了两处，而且都是静默的：

- 全流程串联的 `patience` 默认 50，训练页与启动器都是 100 →
  同一份数据走「一键全流程」和「训练页」早停轮数不同，模型不可比；
- `AnomalibTrainRequest.input_size` 默认 256，而 AnomalyDINO 启动器默认 448 →
  后端把 256 显式拼进命令行，启动器的 448 永远用不上（DINOv2 patch14 切不整齐）。

这类漂移不会报错，只会让「同样的参数跑出不同结果」，所以只能靠测试钉住。

## 三类字段，必须显式登记

1. `HARD`：请求默认值必须**逐键等于**启动器默认值；
2. `FOLLOW`：请求默认 `None` 表示「不传，跟随启动器」——只要求启动器确有这个参数；
3. `EXCEPTIONS`：两边确实不同，但**有正当理由**（下面每条都写了理由）。

第 4 条测试会检查「请求模型的每个字段都被登记过」，所以以后新增字段
（比如新的增强参数）如果忘了登记，这里会直接失败，而不是等到某次实验对不上才发现。

    python 6-web/backend/tools/test_train_defaults.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]

sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "1-python-yolo" / "launcher"))
sys.path.insert(0, str(ROOT / "2-python-anomaly" / "launcher"))

from _harness import case, check, finish, run_cases  # noqa: E402

import tf_anomalib_launcher  # noqa: E402
import tf_anomalydino_launcher  # noqa: E402
import tf_yolo_launcher  # noqa: E402
from routers.pipeline import (  # noqa: E402
    AnomalibPipelineRequest,
    PipelineYoloRequest,
)
from routers.train import (  # noqa: E402
    AnomalibTrainRequest,
    YoloTrainRequest,
)


def launcher_defaults(module, mode: str) -> dict:
    """取启动器在某个 mode 下的 argparse 默认值。"""
    args = module.build_parser().parse_args(["--mode", mode])
    return vars(args)


YOLO = launcher_defaults(tf_yolo_launcher, "train")
ANOMALIB = launcher_defaults(tf_anomalib_launcher, "train-export")
DINO = launcher_defaults(tf_anomalydino_launcher, "build-export")

#: 请求默认值必须与启动器完全一致（字段名 → argparse dest）。
#: 这里每一条都是「用户不填就该等价于命令行不填」的硬约定。
YOLO_HARD = {
    "task": "task",
    "epochs": "epochs",
    "imgsz": "imgsz",
    "batch": "batch",
    "project": "project",
    "patience": "patience",
    "workers": "workers",
    "resume": "resume",
    "exist_ok": "exist_ok",
    "init_mode": "init_mode",
}

#: 请求默认 `None` =「不传这个参数」：启动器自己的默认值生效。
#: 只要求启动器确实有这个 dest，防止拼命令行时拼了一个 argparse 不认识的口令。
YOLO_FOLLOW = {
    "lr0": "lr0",
    "hsv_h": "hsv_h",
    "hsv_s": "hsv_s",
    "hsv_v": "hsv_v",
    "degrees": "degrees",
    "translate": "translate",
    "scale": "scale",
    "shear": "shear",
    "perspective": "perspective",
    "flipud": "flipud",
    "fliplr": "fliplr",
    "bgr": "bgr",
    "mosaic": "mosaic",
    "mixup": "mixup",
    "copy_paste": "copy_paste",
    "close_mosaic": "close_mosaic",
    "amp": "amp",
}

#: 两边不同但**有意为之**。理由必须写清楚，否则下一个人不敢动。
YOLO_EXCEPTIONS = {
    "model": "请求给 yolov8n.pt 做占位；启动器空串是按 task 推导（见 --model 处理），语义相同",
    "device": "启动器 None=自动挑卡；Web 默认固定 0 号卡，台账里的「用了哪张卡」才可复现",
    "name": "启动器 None=按任务名生成；Web 显式 exp，方便用户在页面上改",
    "pretrained": "请求 'true' 与启动器 None（Ultralytics 默认 True）等价，都表示用预训练权重",
    "resume_from": "空串=不续训，两边默认一致，但启动器侧语义是「不指定检查点」",
    "allow_low_vram": "纯 Web 侧策略（显存预检失败是否硬拦），启动器无需知道",
}

#: 不是默认值，而是必填/自由文本字段：没有「默认值」可比。
YOLO_FREE = {"data", "python"}


def as_text(value) -> str:
    """把默认值归一成可比较的字符串。

    启动器里布尔参数是 `"true"/"false"` 字符串（argparse 不认 bool），
    请求模型里是 Python bool；不归一就永远「不相等」。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


@case("YOLO 硬约定默认值逐键相等")
def test_yolo_hard_defaults() -> None:
    fields = YoloTrainRequest.model_fields
    for name, dest in YOLO_HARD.items():
        actual = as_text(fields[name].default)
        expected = as_text(YOLO[dest])
        check(
            f"YoloTrainRequest.{name} == launcher --{dest.replace('_', '-')}（{expected}）",
            actual == expected,
            f"请求默认 {actual!r}，启动器默认 {expected!r}",
        )


@case("YOLO 跟随启动器的字段默认 None 且启动器认得")
def test_yolo_follow_defaults() -> None:
    fields = YoloTrainRequest.model_fields
    for name, dest in YOLO_FOLLOW.items():
        check(
            f"YoloTrainRequest.{name} 默认 None（不传，跟随启动器）",
            fields[name].default is None,
            f"实际默认 {fields[name].default!r}",
        )
        check(
            f"启动器接受 --{dest.replace('_', '-')}",
            dest in YOLO,
            f"argparse 里没有 {dest}",
        )


@case("YOLO 每个字段都登记过")
def test_yolo_fields_classified() -> None:
    known = set(YOLO_HARD) | set(YOLO_FOLLOW) | set(YOLO_EXCEPTIONS) | YOLO_FREE
    unknown = sorted(set(YoloTrainRequest.model_fields) - known)
    check(
        "YoloTrainRequest 的字段全部登记（新增字段必须归类）",
        not unknown,
        f"未登记：{unknown}",
    )
    # 登记了但模型里已经没有的字段，说明清单过期了，同样要清掉
    stale = sorted(known - set(YoloTrainRequest.model_fields))
    check("字段清单没有过期项", not stale, f"多出来：{stale}")


@case("全流程 YOLO 默认值不再与训练页漂移")
def test_pipeline_yolo_defaults() -> None:
    fields = PipelineYoloRequest.model_fields
    for name in ("epochs", "imgsz", "batch", "project", "patience", "workers"):
        actual = as_text(fields[name].default)
        expected = as_text(YOLO[name])
        check(
            f"PipelineYoloRequest.{name} == launcher --{name}（{expected}）",
            actual == expected,
            f"实际 {actual!r}",
        )
    # pretrained 在两个请求模型里都是 "true"（显式传，与启动器 None 等价）
    check(
        "PipelineYoloRequest.pretrained 与训练页同为 'true'",
        as_text(fields["pretrained"].default)
        == as_text(YoloTrainRequest.model_fields["pretrained"].default)
        == "true",
        f"实际 {fields['pretrained'].default!r}",
    )
    # name 留空是「自动生成 pipe_<时间戳>」，与启动器的「按任务名生成」不同，登记在此
    check(
        "PipelineYoloRequest.name 默认空串（自动生成带时间戳的名字）",
        fields["name"].default == "",
        f"实际 {fields['name'].default!r}",
    )


@case("Anomalib 硬约定默认值逐键相等")
def test_anomalib_defaults() -> None:
    fields = AnomalibTrainRequest.model_fields
    hard = {
        "model": "model",
        "normal_dir": "normal_dir",
        "abnormal_dir": "abnormal_dir",
        "normal_test_dir": "normal_test_dir",
        "project": "project",
        "name": "name",
        "max_epochs": "max_epochs",
        "train_batch_size": "train_batch_size",
        "eval_batch_size": "eval_batch_size",
        "num_workers": "num_workers",
        "val_split_ratio": "val_split_ratio",
        "calib_limit": "calib_limit",
        "imagenet_dir": "imagenet_dir",
        "calib_normal_dir": "calib_normal_dir",
        "calib_abnormal_dir": "calib_abnormal_dir",
        "calibrate_metadata": "calibrate_metadata",
    }
    for name, dest in hard.items():
        actual = as_text(fields[name].default)
        expected = as_text(ANOMALIB[dest])
        check(
            f"AnomalibTrainRequest.{name} == anomalib launcher --{dest.replace('_', '-')}",
            actual == expected,
            f"请求默认 {actual!r}，启动器默认 {expected!r}",
        )

    # input_size 是「按模型分流」的：不填 Anomalib 256，不填 AnomalyDINO 448
    check(
        "AnomalibTrainRequest.input_size 不填且模型是 patchcore 时取 256",
        AnomalibTrainRequest(root="x").input_size == ANOMALIB["input_size"] == 256,
        f"实际 {AnomalibTrainRequest(root='x').input_size}",
    )
    check(
        "AnomalibTrainRequest.input_size 不填且模型是 anomalydino 时取 448",
        AnomalibTrainRequest(root="x", model="anomalydino").input_size == DINO["input_size"] == 448,
        f"实际 {AnomalibTrainRequest(root='x', model='anomalydino').input_size}",
    )
    check(
        "显式传 input_size 不被按模型改写（用户要 256 就是 256）",
        AnomalibTrainRequest(root="x", model="anomalydino", input_size=256).input_size == 256,
        "显式值被覆盖了",
    )


@case("AnomalyDINO 专属默认值与零样本启动器一致")
def test_anomalydino_defaults() -> None:
    request = AnomalibTrainRequest(root="x", model="anomalydino")
    check(
        f"shots 默认 {DINO['shots']}",
        request.shots == DINO["shots"],
        f"实际 {request.shots}",
    )
    check(
        f"rotation 默认 {DINO['rotation']}",
        as_text(request.rotation) == as_text(DINO["rotation"]),
        f"实际 {request.rotation!r}",
    )
    check(f"seed 默认 {DINO['seed']}", request.seed == DINO["seed"], f"实际 {request.seed}")
    # 零样本不划训练/验证集：请求统一发 0.2，但零样本启动器忽略该参数。
    # 这不是漂移，登记在此以免以后有人照着「对齐」改成 0.0 又说不清原因。
    check(
        "零样本启动器的 val_split_ratio 是 0.0（请求统一发 0.2，启动器忽略）",
        DINO["val_split_ratio"] == 0.0,
        f"实际 {DINO['val_split_ratio']!r}",
    )


@case("全流程异常检测默认值与训练请求一致")
def test_pipeline_anomaly_defaults() -> None:
    fields = AnomalibPipelineRequest.model_fields
    hard = {
        "model": "model",
        "normal_dir": "normal_dir",
        "abnormal_dir": "abnormal_dir",
        "normal_test_dir": "normal_test_dir",
        "project": "project",
        "max_epochs": "max_epochs",
        "train_batch_size": "train_batch_size",
        "eval_batch_size": "eval_batch_size",
        "num_workers": "num_workers",
        "val_split_ratio": "val_split_ratio",
        "calib_limit": "calib_limit",
        "imagenet_dir": "imagenet_dir",
        "calib_normal_dir": "calib_normal_dir",
        "calib_abnormal_dir": "calib_abnormal_dir",
        "calibrate_metadata": "calibrate_metadata",
    }
    for name, dest in hard.items():
        actual = as_text(fields[name].default)
        expected = as_text(ANOMALIB[dest])
        check(
            f"AnomalibPipelineRequest.{name} == anomalib launcher --{dest.replace('_', '-')}",
            actual == expected,
            f"请求默认 {actual!r}，启动器默认 {expected!r}",
        )
    check(
        "全流程 anomalydino 不填 input_size 时也是 448",
        AnomalibPipelineRequest(model="anomalydino").input_size == 448,
        f"实际 {AnomalibPipelineRequest(model='anomalydino').input_size}",
    )
    # name 留空=自动生成 pipe_<时间戳>，与训练页/启动器的 latest 不同（有意）
    check(
        "AnomalibPipelineRequest.name 默认空串（自动生成带时间戳的名字）",
        fields["name"].default == "",
        f"实际 {fields['name'].default!r}",
    )


def main() -> int:
    run_cases(globals())
    return finish("训练默认值护栏")


if __name__ == "__main__":
    raise SystemExit(main())
