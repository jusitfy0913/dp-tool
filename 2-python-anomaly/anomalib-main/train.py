import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6+ 默认 torch.load(..., weights_only=True)；Lightning/anomalib 的 ckpt
# 含自定义类与 pathlib 等，拒载时导出 ONNX 会失败。训练导出场景仅加载本地可信权重。
_orig_torch_load = torch.load


def _torch_load_trusted_ckpt(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted_ckpt

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from anomalib.models import EfficientAd, Patchcore
from anomalib.models.image.padim import Padim
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.pipelines.benchmark import Benchmark
from anomalib.deploy.export import ExportType
from trtforge_metadata import (
    apply_postprocessed_output_contract,
    calibrate_metadata_from_onnx,
    extract_model_metadata,
    onnx_has_postprocessed_outputs,
    parse_anomaly_outputs,
    preprocess_image_for_onnx,
    write_metadata_json,
)

# 与 Folder 数据分辨率一致，便于 ONNX 静态导出（Patchcore 导出需要）
EXPORT_INPUT_SIZE = (256, 256)

MODEL_FACTORIES = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficientad": EfficientAd,
}

# 演示数据集根目录。
# 默认取脚本同级的 anomalib-main/datasets/bottle（仓库相对推导，不写死机器绝对路径）；
# 需要指向其它数据集时设置环境变量 TRTFORGE_ANOMALIB_DATASET_ROOT。
DATASET_ROOT = Path(
    os.environ.get("TRTFORGE_ANOMALIB_DATASET_ROOT", "").strip()
    or (Path(__file__).resolve().parent / "datasets" / "bottle")
).resolve()

# 数据集子目录约定：train/good 为正样本训练集，test/good 与 test/abnormal 为测试集
DATASET_TRAIN_GOOD = DATASET_ROOT / "train" / "good"
DATASET_TEST_GOOD = DATASET_ROOT / "test" / "good"
DATASET_TEST_ABNORMAL = DATASET_ROOT / "test" / "abnormal"


def normalize_model_name(model_name: str) -> str:
    """Normalize user-facing model names to the internal factory key."""
    key = model_name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key not in MODEL_FACTORIES:
        supported = ", ".join(MODEL_FACTORIES)
        raise argparse.ArgumentTypeError(f"不支持的模型: {model_name}，可选: {supported}")
    return key


def model_name_from_input_args(input_args: list[str] | None, default: str) -> str:
    """兼容旧入口的 --InputArgs model=xxx / --model xxx 写法。"""
    if not input_args:
        return default
    for index, value in enumerate(input_args):
        if value in {"--model", "model"} and index + 1 < len(input_args):
            return normalize_model_name(input_args[index + 1])
        if value.startswith("--model="):
            return normalize_model_name(value.split("=", 1)[1])
        if value.startswith("model="):
            return normalize_model_name(value.split("=", 1)[1])
    return default


def create_model(model_name: str):
    """Create the selected Anomalib model."""
    if model_name == "efficientad":
        # 固定到脚本所在目录，避免从其它工作目录启动时相对路径指向错误的 datasets
        imagenet_dir = Path(__file__).resolve().parent / "datasets" / "imagenette"
        return EfficientAd(imagenet_dir=imagenet_dir)
    return MODEL_FACTORIES[model_name]()


def ensure_demo_folder_images() -> None:
    """若演示数据集为空则写入最小 PNG，避免训练/导出链路在无数据时报错。"""
    root = DATASET_ROOT
    plan = [("train/good", 16), ("test/good", 8), ("test/abnormal", 8)]
    from PIL import Image
    import numpy as np

    for sub, n in plan:
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        if any(d.glob("*.png")):
            continue
        for i in range(n):
            rng = np.random.default_rng(42 + i + hash(sub) % 1000)
            arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            if "abnormal" in sub:
                arr[80:180, 80:180] = np.array([255, 32, 32], dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img_{i:03d}.png")


def print_batch_fields(batch):
    """Print current anomalib batch fields for quick dataloader verification."""
    if hasattr(batch, "keys"):
        print(batch.keys())
    elif hasattr(batch, "__dataclass_fields__"):
        print(batch.__dataclass_fields__.keys())
    else:
        print(type(batch))


def collect_calibration_images(root: Path, limit: int = 32) -> list[Path]:
    """Collect a small deterministic image set for ONNX metadata calibration."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in suffixes]
    return paths[:limit]


def _percentile(values: list[float] | np.ndarray, percentile: float) -> float:
    """Return a stable percentile value for threshold calibration."""
    if len(values) == 0:
        raise ValueError("percentile input is empty")
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def _threshold_from_distributions(normal_values: list[float], abnormal_values: list[float]) -> float:
    """Choose a threshold that keeps normal samples below the NG boundary."""
    normal_high = _percentile(normal_values, 99.0)
    if not abnormal_values:
        return normal_high

    abnormal_low = _percentile(abnormal_values, 5.0)
    if abnormal_low > normal_high:
        return (normal_high + abnormal_low) / 2.0

    all_values = normal_values + abnormal_values
    span = max(max(all_values) - min(all_values), 1.0e-6)
    return normal_high + span * 0.02


def calibrate_efficientad_metadata_from_onnx(
    onnx_path: Path,
    normal_roots: list[Path],
    abnormal_roots: list[Path],
    metadata: dict,
    model_type: str,
    limit_per_group: int = 64,
) -> dict:
    """Calibrate EfficientAd thresholds from normal/abnormal ONNX output distributions."""
    normal_images: list[Path] = []
    for root in normal_roots:
        normal_images.extend(collect_calibration_images(root, limit=limit_per_group))
    abnormal_images: list[Path] = []
    for root in abnormal_roots:
        abnormal_images.extend(collect_calibration_images(root, limit=limit_per_group))

    normal_images = normal_images[:limit_per_group]
    abnormal_images = abnormal_images[:limit_per_group]
    if not normal_images:
        print("EfficientAd metadata calibration skipped: normal images not found")
        return metadata

    try:
        import onnxruntime as ort
    except ImportError:
        print("EfficientAd metadata calibration skipped: onnxruntime is not installed")
        return metadata

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    normal_scores: list[float] = []
    abnormal_scores: list[float] = []
    normal_pixels: list[float] = []
    abnormal_pixels: list[float] = []
    map_mins: list[float] = []
    map_maxs: list[float] = []

    def infer_group(paths: list[Path], scores: list[float], pixels: list[float]) -> None:
        for path in paths:
            tensor = preprocess_image_for_onnx(path, metadata, model_type)
            outputs = session.run(None, {input_name: tensor})
            score, anomaly_map = parse_anomaly_outputs(outputs, output_names)
            scores.append(float(score))
            if anomaly_map is None:
                continue
            arr = np.asarray(anomaly_map, dtype=np.float32)
            map_mins.append(float(np.min(arr)))
            map_maxs.append(float(np.max(arr)))
            pixels.append(_percentile(arr.reshape(-1), 99.5))

    infer_group(normal_images, normal_scores, normal_pixels)
    infer_group(abnormal_images, abnormal_scores, abnormal_pixels)
    if not normal_scores:
        print("EfficientAd metadata calibration skipped: ONNX normal scores are empty")
        return metadata

    all_scores = normal_scores + abnormal_scores
    score_threshold = _threshold_from_distributions(normal_scores, abnormal_scores)
    pixel_threshold = (
        _threshold_from_distributions(normal_pixels, abnormal_pixels)
        if normal_pixels
        else score_threshold
    )

    metadata["output_contract"] = "efficientad_distribution_calibrated"
    metadata["normalization"] = "none"
    metadata["image_threshold"] = score_threshold
    metadata["pixel_threshold"] = pixel_threshold
    metadata["pred_scores"]["min"] = min(all_scores)
    metadata["pred_scores"]["max"] = max(max(all_scores), score_threshold)
    metadata["pred_scores.min"] = metadata["pred_scores"]["min"]
    metadata["pred_scores.max"] = metadata["pred_scores"]["max"]
    if map_mins and map_maxs:
        metadata["anomaly_maps"]["min"] = min(map_mins)
        metadata["anomaly_maps"]["max"] = max(max(map_maxs), pixel_threshold)
        metadata["anomaly_maps.min"] = metadata["anomaly_maps"]["min"]
        metadata["anomaly_maps.max"] = metadata["anomaly_maps"]["max"]
    metadata["calibration"] = {
        "source": "onnxruntime_efficientad_distribution",
        "normal_image_count": len(normal_images),
        "abnormal_image_count": len(abnormal_images),
        "normal_score_min": min(normal_scores),
        "normal_score_max": max(normal_scores),
        "abnormal_score_min": min(abnormal_scores) if abnormal_scores else None,
        "abnormal_score_max": max(abnormal_scores) if abnormal_scores else None,
    }
    print(
        "EfficientAd metadata calibrated: "
        f"image_threshold={score_threshold:.6f}, pixel_threshold={pixel_threshold:.6f}"
    )
    return metadata


def train(model_name: str = "patchcore"):
    # Benchmark().run()
    # dataset:
    #   name: tube               # 数据集的名字，如MVTec等，这个不重要
    #   format: folder
    #   path: ./datasets/img_192 # 自制数据集路径
    #   normal_dir: normal       # 自制数据集正样本子文件夹
    #   abnormal_dir: abnormal   # 自制数据集负样本子文件夹
    #   mask_dir: null           # 二值掩膜路径，自制数据集一般没有，填null
    #   normal_test_dir: null # name of the folder containing normal test images.
    #   task: classification # classification or segmentation
    #   extensions: null
    #   normalization: imagenet  # 此处添加imagenet
    #   split_ratio: 0.2 # ratio of the normal images that will be used to create a test split
    #   image_size: 256
    #   train_batch_size: 32
    #   test_batch_size: 32
    #   num_workers: 8
    #   transform_config:
    #     train: null
    #     val: null
    #   test_split_mode: from_dir # 此处添加
    #   test_split_ratio: 0.2
    #   val_split_mode: same_as_test
    #   val_split_ratio: 0.5
    #   create_validation_set: true
    #   tiling:
    #     apply: false
    #     tile_size: null
    #     stride: null
    #     remove_border_count: 0
    #     use_random_tiling: False
    #     random_tile_count: 16
    ensure_demo_folder_images()
    print("----------------------------1-------------------------------")
    # EfficientAd 在 anomalib 中强制要求 train_batch_size == 1，否则会在 on_train_start 报错
    train_batch_size = 1 if model_name == "efficientad" else 4
    if model_name == "efficientad":
        print(f"train_batch_size={train_batch_size}（EfficientAd 训练 batch 必须为 1）")
    else:
        print(f"train_batch_size={train_batch_size}")
    datamodule = Folder(
        name="demo_5",
        root=str(DATASET_ROOT),
        normal_dir=str(DATASET_TRAIN_GOOD),
        abnormal_dir=str(DATASET_TEST_ABNORMAL),
        normal_test_dir=str(DATASET_TEST_GOOD),
        num_workers=0,
        val_split_ratio=0.2,
        train_batch_size=train_batch_size,
    )
    # datamodule = Folder(
    #     name="demo_1",
    #     root=r"E:\mvtec_anomaly_detection.tar\capsule",
    #     normal_dir=r"E:\mvtec_anomaly_detection.tar\capsule\train\good",
    #     abnormal_dir=r"E:\mvtec_anomaly_detection.tar\capsule\test\squeeze",
    #     normal_test_dir=r"E:\mvtec_anomaly_detection.tar\capsule\test\good",
    #     num_workers=0, 
    #     val_split_ratio=0.5,
    #     train_batch_size = 1,
    # )
    # name (str): Name of the dataset. Used for logging/saving.
    #         normal_dir (str | Path | Sequence): Directory containing normal images.
    #         root (str | Path | None): Root folder containing normal and abnormal
    #             directories. Defaults to ``None``.
    #         abnormal_dir (str | Path | None | Sequence): Directory containing
    #             abnormal images. Defaults to ``None``.
    #         normal_test_dir (str | Path | Sequence | None): Directory containing
    #             normal test images. Defaults to ``None``.
    #         mask_dir (str | Path | Sequence | None): Directory containing mask
    #             annotations. Defaults to ``None``.
    #         normal_split_ratio (float): Ratio to split normal training images for
    #             test set when no normal test images exist.
    #             Defaults to ``0.2``.
    #         extensions (tuple[str, ...] | None): Image extensions to include.
    #             Defaults to ``None``.
    #         train_batch_size (int): Training batch size.
    #             Defaults to ``32``.
    #         eval_batch_size (int): Validation/test batch size.
    #             Defaults to ``32``.
    #         num_workers (int): Number of workers for data loading.
    #             Defaults to ``8``.
    #         train_augmentations (Transform | None): Augmentations to apply dto the training images
    #             Defaults to ``None``.
    #         val_augmentations (Transform | None): Augmentations to apply to the validation images.
    #             Defaults to ``None``.
    #         test_augmentations (Transform | None): Augmentations to apply to the test images.
    #             Defaults to ``None``.
    #         augmentations (Transform | None): General augmentations to apply if stage-specific
    #             augmentations are not provided.
    #         test_split_mode (TestSplitMode): Method to obtain test subset.
    #             Defaults to ``TestSplitMode.FROM_DIR``.
    #         test_split_ratio (float): Fraction of train images for testing.
    #             Defaults to ``0.2``.
    #         val_split_mode (ValSplitMode): Method to obtain validation subset.
    #             Defaults to ``ValSplitMode.FROM_TEST``.
    #         val_split_ratio (float): Fraction of images for validation.
    #             Defaults to ``0.5``.
    #         seed (int | None): Random seed for splitting.
    #             Defaults to ``None``.
    # Setup the datamodule
    print("----------------------------2-------------------------------")
    datamodule.setup()

    _, train_data = next(enumerate(datamodule.train_dataloader()))
    print_batch_fields(train_data)

    _, val_data = next(enumerate(datamodule.val_dataloader()))
    print_batch_fields(val_data)

    _, test_data = next(enumerate(datamodule.test_dataloader()))
    print_batch_fields(test_data)
    print("----------------------------3-------------------------------")
    model = create_model(model_name)
    print(f"selected model: {model.__class__.__name__}")
    print("----------------------------4-------------------------------")
    # PaDiM 仅需 1 个 epoch（估计均值协方差）；EfficientAd 等需多轮
    if model_name == "padim":
        train_max_epochs = 1
    elif model_name == "efficientad":
        train_max_epochs = 5
    else:
        train_max_epochs = 50
    print(f"max_epochs={train_max_epochs}")
    # Lightning CSVLogger 在多 epoch 且指标键随步数变化时可能重写 metrics.csv 报错
    # （EfficientAd 等会混用 step/epoch 指标名）。多轮训练关闭默认 CSV，单 epoch 仍可记录。
    use_logger = train_max_epochs <= 1
    if not use_logger:
        print("logger=False（多 epoch 时跳过 CSVLogger，避免 metrics.csv 表头冲突）")
    # Windows 控制台默认 GBK 时，Lightning 的 Rich 进度条 teardown 可能触发 UnicodeEncodeError
    engine = Engine(max_epochs=train_max_epochs, logger=use_logger, enable_progress_bar=False)
    # engine = Engine(max_epochs=2,
    #                 logger = True)
    # engine = Engine()
    engine.train(datamodule=datamodule, model=model)

    print("----------------------------5 export ONNX-------------------------------")
    export_root = Path(engine.trainer.default_root_dir)
    ckpt_dir = export_root / "weights" / "lightning"
    ckpt_path = ckpt_dir / "model.ckpt"
    if not ckpt_path.is_file():
        cb = engine.checkpoint_callback
        if cb is not None and getattr(cb, "dirpath", None):
            alt = Path(cb.dirpath) / (getattr(cb, "filename", "model") + ".ckpt")
            if alt.is_file():
                ckpt_path = alt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 Lightning checkpoint: {ckpt_path}")

    onnx_path = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=export_root,
        ckpt_path=ckpt_path,
        input_size=EXPORT_INPUT_SIZE,
    )
    if onnx_path is None:
        raise RuntimeError("ONNX 导出失败（返回路径为空）")
    onnx_path = Path(onnx_path)
    target_onnx = ckpt_dir / onnx_path.name
    if onnx_path.resolve() != target_onnx.resolve():
        shutil.move(str(onnx_path), str(target_onnx))
        onnx_path = target_onnx
    metadata = extract_model_metadata(
        model=model,
        model_type=model.__class__.__name__,
        input_size=EXPORT_INPUT_SIZE,
    )
    if model_name == "efficientad":
        metadata = calibrate_efficientad_metadata_from_onnx(
            onnx_path=onnx_path,
            normal_roots=[DATASET_TRAIN_GOOD, DATASET_TEST_GOOD],
            abnormal_roots=[DATASET_TEST_ABNORMAL],
            metadata=metadata,
            model_type=model.__class__.__name__,
        )
    elif onnx_has_postprocessed_outputs(onnx_path):
        metadata = apply_postprocessed_output_contract(metadata)
    else:
        calibration_images = collect_calibration_images(DATASET_ROOT / "test")
        metadata = calibrate_metadata_from_onnx(
            onnx_path=onnx_path,
            image_paths=calibration_images,
            metadata=metadata,
            model_type=model.__class__.__name__,
        )
    metadata_path = write_metadata_json(
        output_path=ckpt_dir / "metadata.json",
        metadata=metadata,
        ckpt_path=ckpt_path,
        onnx_path=onnx_path,
    )
    print(f"checkpoint: {ckpt_path.resolve()}")
    print(f"onnx      : {onnx_path.resolve()}")
    print(f"metadata  : {metadata_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="patchcore",
        type=normalize_model_name,
        help="选择 Anomalib 模型：patchcore、padim、efficientad，默认 patchcore。",
    )
    parser.add_argument("--InputArgs", nargs="*", default=None)
    args = parser.parse_args()
    selected_model = model_name_from_input_args(args.InputArgs, args.model)
    print(f"InputArgs: {args.InputArgs}")
    train(model_name=selected_model)
