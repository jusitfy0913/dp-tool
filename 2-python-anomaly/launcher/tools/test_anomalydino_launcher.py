# -*- coding: utf-8 -*-
"""AnomalyDINO 启动器的纯逻辑验收，不需要 GPU、不需要 trtexec。

守的是「导出的图能编译、能推理，但结果不是 AnomalyDINO」这一类问题：

- 位置编码用 `size=(grid, grid)` 插值 → 与上游逐 token 余弦只有 0.99（必须用 scale_factor）；
- 记忆库忘了 L2 归一 → 余弦距离恒等于 1，热力图全白；
- 多导出一个输出，且元素数比异常图大 → `anomalib_infer.dll` 会把它当异常图（静默错）；
- 正常图不够还继续建库 → 判定随机，日志却说成功；
- metadata 少写一个点号平铺键 → DLL 在 `ai_create` 阶段直接失败。

真实权重在这台机器上存在时会额外跑一遍主干，验证 448 的网格是 32×32。

    python 2-python-anomaly/launcher/tools/test_anomalydino_launcher.py
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAUNCHER_DIR))

import anomalydino_core as core  # noqa: E402
import tf_anomalydino_launcher as launcher  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


class FakeBackbone:
    """把输入压成固定 token 的假主干：只验证算法与导出，不碰 timm。"""

    def __init__(self, cells: int, dim: int = 384, seed: int = 0) -> None:
        import torch

        generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.randn(1, cells, dim, generator=generator)
        self.calls = 0

    def forward_features(self, batch):
        import torch

        self.calls += 1
        # 必须真的用到输入：完全不依赖输入的话，torch.onnx.export 会把 graph.input 剪成空，
        # 导出的模型没有入口，编译时才会暴露。这里用一个标量缩放制造依赖。
        scale = 1.0 + batch.mean()
        return torch.cat(
            [torch.zeros(batch.shape[0], 1, self.tokens.shape[-1]), self.tokens * scale], dim=1
        )

    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self


def read_events(text: str) -> list[dict]:
    """从混合输出里挑出 `TRTFORGE_JSON` 事件行。"""
    events = []
    for line in text.splitlines():
        match = re.search(r"TRTFORGE_JSON\s*(\{.*\})", line)
        if match:
            events.append(json.loads(match.group(1)))
    return events


def run_main(argv: list[str]) -> tuple[int, list[dict], str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = launcher.main(argv)
    text = buffer.getvalue()
    return code, read_events(text), text


def main() -> int:
    import numpy as np
    import torch

    tmp = Path(tempfile.mkdtemp(prefix="anomalydino_test_"))
    try:
        # ---- 1. 常量与契约 ---------------------------------------------------
        check("模型名就是 metadata 里的 model_type",
              launcher.MODEL_LABEL == "AnomalyDINO", launcher.MODEL_LABEL)
        check("输出契约与 DLL 的解析分支对应",
              launcher.OUTPUT_CONTRACT == "trtforge_anomalydino_v1", launcher.OUTPUT_CONTRACT)
        check("三个模式齐全",
              launcher.SUPPORTED_MODES == {"check-env", "build-export", "warm"},
              str(launcher.SUPPORTED_MODES))

        # ---- 2. 网格换算 -----------------------------------------------------
        check("448 → 32×32 网格", core.grid_for_resolution(448) == (32, 32),
              str(core.grid_for_resolution(448)))
        check("224 → 16×16 网格", core.grid_for_resolution(224) == (16, 16))
        try:
            core.grid_for_resolution(256)
            check("256 不是 14 的倍数要报错", False)
        except ValueError as exc:
            check("256 不是 14 的倍数要报错", "14" in str(exc), str(exc))

        # ---- 3. 位置编码插值 -------------------------------------------------
        base = 37
        fake_pos = torch.randn(1, 1 + base * base, 384)
        out = core.build_pos_embed(fake_pos, 32, 32)
        check("pos_embed 长度 = 1 + 32*32", tuple(out.shape) == (1, 1 + 1024, 384),
              str(tuple(out.shape)))
        check("cls token 的位置编码原样保留",
              torch.equal(out[:, :1], fake_pos[:, :1]))
        try:
            core.build_pos_embed(torch.randn(1, 1 + 10, 384), 4, 4)
            check("非平方数 patch 要报错", False)
        except ValueError:
            check("非平方数 patch 要报错", True)
        try:
            core.build_pos_embed(torch.randn(2, 1 + 4, 384), 2, 2)
            check("batch 维度不为 1 要报错", False)
        except ValueError:
            check("batch 维度不为 1 要报错", True)

        # ---- 4. 记忆库行数 = 图数 × 网格单元数 --------------------------------
        cells = 1024
        tokens = [torch.randn(cells, 384) for _ in range(3)]
        bank = core.build_bank(tokens)
        check("库行数 = 3 张 × 1024 格", bank.shape == (3 * cells, 384), str(tuple(bank.shape)))
        check("库每行都是单位向量",
              torch.allclose(bank.norm(dim=-1), torch.ones(bank.shape[0])), "")
        try:
            core.build_bank([])
            check("空库要报错", False)
        except ValueError:
            check("空库要报错", True)
        check("同一批 token 的指纹稳定", core.bank_digest(bank) == core.bank_digest(bank.clone()))
        check("换顺序指纹就变",
              core.bank_digest(bank) != core.bank_digest(torch.roll(bank, 1, dims=0)))
        check("top-1% 向下取整但不为 0", core.top1p_count(1024) == 10, str(core.top1p_count(1024)))
        check("格子很少时至少取 1 个", core.top1p_count(16) == 1, str(core.top1p_count(16)))

        # ---- 5. 固化库与参考实现逐元素一致 ----------------------------------
        grid = 4
        facade = FakeBackbone(cells=grid * grid)
        small_bank = core.build_bank([torch.randn(grid * grid, 384) for _ in range(2)])
        wrapper = core.BakedAnomalyDINO(facade, small_bank, grid, grid, chunk_rows=8)
        image = torch.zeros(1, 3, 56, 56)
        anomaly_map, score = wrapper(image)
        ref_map, ref_score = core.predict_reference(
            facade, image, small_bank, grid, grid, chunk_rows=8
        )
        check("异常图形状 [1,1,4,4]", tuple(anomaly_map.shape) == (1, 1, 4, 4),
              str(tuple(anomaly_map.shape)))
        check("分数形状 [1,1]", tuple(score.shape) == (1, 1), str(tuple(score.shape)))
        check("固化库与参考实现的热力图一致", torch.allclose(anomaly_map, ref_map, atol=1e-6))
        check("固化库与参考实现的分数一致", torch.allclose(score, ref_score, atol=1e-6))
        check("热力图取值在 [0, 2] 内",
              float(anomaly_map.min()) >= -1e-6 and float(anomaly_map.max()) <= 2 + 1e-6,
              f"{float(anomaly_map.min())}, {float(anomaly_map.max())}")
        check("分块大小不影响结果",
              torch.allclose(
                  wrapper(image)[0],
                  core.BakedAnomalyDINO(facade, small_bank, grid, grid, chunk_rows=5)(image)[0],
              ))
        check("库是持久化 buffer（导出时才会变成常量）",
              "bank" in dict(wrapper.named_buffers()))

        # ---- 6. 导出契约：只允许两个输出，且异常图最大 ----------------------
        onnx_path = tmp / "fake.onnx"
        launcher.export_onnx(wrapper, 56, onnx_path)
        import onnx

        model = onnx.load(str(onnx_path))
        output_names = [node.name for node in model.graph.output]
        input_shape = [
            dim.dim_value if dim.HasField("dim_value") else -1
            for dim in model.graph.input[0].type.tensor_type.shape.dim
        ]
        check("ONNX 只有两个输出", output_names == ["anomaly_map", "pred_score"],
              str(output_names))
        check("输入名固定为 input", model.graph.input[0].name == "input")
        check("输入形状静态 [1,3,56,56]", input_shape == [1, 3, 56, 56], str(input_shape))
        check("opset 固定 16（TRT 8.5 没有 LayerNormalization 插件）",
              any(op.version == 16 for op in model.opset_import), str(model.opset_import))
        sizes = []
        for node in model.graph.output:
            dims = [d.dim_value for d in node.type.tensor_type.shape.dim]
            sizes.append(int(np.prod(dims)))
        check("异常图的元素数最多（DLL 的挑选规则）", sizes[0] > sizes[1], str(sizes))

        session, input_name, names = launcher.open_onnx_session(onnx_path)
        raw = session.run(None, {input_name: image.numpy()})
        onnx_score, onnx_map = launcher.parse_outputs(raw, names)
        check("ONNX 与 PyTorch 的分数一致",
              abs(onnx_score - float(score)) < 1e-4, f"{onnx_score} vs {float(score)}")
        check("ONNX 与 PyTorch 的热力图相关 ≈ 1",
              float(np.corrcoef(np.ravel(onnx_map), np.ravel(anomaly_map.numpy()))[0, 1]) > 0.999,
              "")

        # ---- 7. 阈值规则 -----------------------------------------------------
        normal = [0.1, 0.12, 0.15, 0.2, 0.22]
        threshold_only_normal = launcher.threshold_from_distributions(normal, [])
        check("只有正常图时用 99 分位", abs(threshold_only_normal - launcher.percentile(normal, 99.0)) < 1e-9,
              str(threshold_only_normal))
        separated = launcher.threshold_from_distributions(normal, [0.8, 0.9, 1.0])
        check("分布分得开时取中间", 0.2 < separated < 0.8, str(separated))
        overlapping = launcher.threshold_from_distributions(normal, [0.05, 0.08])
        check("分布重叠时略高于正常上界",
              overlapping >= launcher.percentile(normal, 99.0), str(overlapping))
        worse = launcher.threshold_from_distributions(normal, [0.3, 0.4, 0.5])
        better = launcher.threshold_from_distributions(normal, [1.0, 1.1, 1.2])
        check("异常越明显阈值越高（单调）", better > worse, f"{better} <= {worse}")

        # ---- 8. 输出挑选规则与 DLL 一致 --------------------------------------
        score_a, map_a = launcher.parse_outputs(
            [np.zeros((1, 1, 32, 32), np.float32), np.array([[0.7]], np.float32)],
            ["anomaly_map", "pred_score"],
        )
        check("按名字挑出分数", abs(score_a - 0.7) < 1e-6, str(score_a))
        check("按名字挑出异常图", map_a.shape == (1, 1, 32, 32), str(map_a.shape))
        score_b, map_b = launcher.parse_outputs(
            [np.zeros((1, 1, 8, 8), np.float32), np.zeros((1, 1, 64, 64), np.float32)],
            ["anomaly_map", "raw_mask"],
        )
        check("元素数更大的输出会顶替异常图（与 DLL 一致）",
              map_b.shape == (1, 1, 64, 64), str(map_b.shape))
        score_c, map_c = launcher.parse_outputs([np.zeros((1, 1, 8, 8), np.float32)], [])
        check("没有分数输出时取热力图最大值", score_c == 0.0, str(score_c))

        # ---- 9. metadata 字段齐全 + 点号平铺键 --------------------------------
        meta_path = tmp / "metadata.json"
        import copy

        metadata = {
            "image_threshold": 0.51,
            "pixel_threshold": 0.62,
            "pred_scores": {"min": 0.1, "max": 1.2},
            "anomaly_maps": {"min": 0.05, "max": 1.4},
            "calibration": {"source": "test", "normal_image_count": 3, "abnormal_image_count": 2},
        }
        stats = {"rows": 3072, "images": 3, "augmented": 3, "sha256": "deadbeef"}
        meta = launcher.write_metadata(
            meta_path, metadata, onnx_path, 448, stats, 16, False, 0,
            Path("dinov2_vits14_pretrain.pth"), "abc123",
        )
        payload = json.loads(Path(meta).read_text(encoding="utf-8"))
        for key in ("model_type", "output_contract", "normalization", "input_size",
                    "backbone", "backbone_sha256", "patch_size", "grid", "shots",
                    "bank_rows", "bank_sha256", "onnx",
                    "image_threshold", "pixel_threshold", "pred_scores", "anomaly_maps"):
            check(f"metadata 含 {key}", key in payload, str(list(payload)))
        for key, want in (("pred_scores.min", 0.1), ("pred_scores.max", 1.2),
                          ("anomaly_maps.min", 0.05), ("anomaly_maps.max", 1.4)):
            check(f"metadata 含平铺键 {key}", abs(float(payload.get(key, -1)) - want) < 1e-9,
                  str(payload.get(key)))
        check("metadata 的 model_type 是 AnomalyDINO",
              payload["model_type"] == "AnomalyDINO", payload["model_type"])
        check("metadata 的 normalization 是 imagenet",
              payload["normalization"] == "imagenet", payload["normalization"])
        check("metadata 的输入尺寸是 448", payload["input_size"] == [448, 448],
              str(payload["input_size"]))
        check("metadata 的网格是 32×32", payload["grid"] == [32, 32], str(payload["grid"]))
        # 上界必须覆盖阈值，否则 DLL 归一化后阈值会跑到 1.0 之外
        check("分数上界不低于阈值", payload["pred_scores"]["max"] >= payload["image_threshold"])
        check("热力图上界不低于阈值", payload["anomaly_maps"]["max"] >= payload["pixel_threshold"])
        copy.deepcopy(payload)

        # ---- 10. 错误路径 ----------------------------------------------------
        empty = tmp / "empty"
        empty.mkdir()
        try:
            launcher.validate_image_dir(empty, "正常图像目录")
            check("空目录要报错", False)
        except FileNotFoundError:
            check("空目录要报错", True)
        missing = tmp / "not_an_image.txt"
        missing.write_text("x", encoding="utf-8")
        not_image_dir = tmp / "only_text"
        not_image_dir.mkdir()
        (not_image_dir / "a.txt").write_text("x", encoding="utf-8")
        try:
            launcher.validate_image_dir(not_image_dir, "正常图像目录")
            check("只有非图片文件的目录要报错", False)
        except FileNotFoundError:
            check("只有非图片文件的目录要报错", True)

        original_candidates = core.default_checkpoint_candidates
        try:
            core.default_checkpoint_candidates = lambda explicit=None: [tmp / "nope.pth"]
            try:
                core.resolve_checkpoint()
                check("缺权重要报错", False)
            except FileNotFoundError as exc:
                check("缺权重要报错", "pre_trained" in str(exc), str(exc))
        finally:
            core.default_checkpoint_candidates = original_candidates

        import cv2

        def make_dataset(count: int) -> Path:
            root = tmp / f"dataset_{count}"
            normal = root / "train" / "good"
            normal.mkdir(parents=True)
            for index in range(count):
                image = np.full((48, 64, 3), index * 5 % 255, dtype=np.uint8)
                cv2.imwrite(str(normal / f"g{index}.png"), image)
            return root

        # 分辨率不合法：属于「参数/数据问题」，按约定用退出码 2（1 留给崩溃与缺依赖）
        root = make_dataset(8)
        code, events, text = run_main(
            ["--mode", "build-export", "--root", str(root), "--input-size", "250"]
        )
        check("非 14 倍数的输入尺寸退出码 2", code == 2, str(code))
        check("非 14 倍数的输入尺寸给出 UNSUPPORTED_RESOLUTION",
              any(event.get("code") == "UNSUPPORTED_RESOLUTION" for event in events),
              text[-300:])

        # 正常图不足：shots 抽到的张数少于下限
        code, events, text = run_main(
            ["--mode", "build-export", "--root", str(root), "--shots", "4", "--device", "cpu"]
        )
        check("正常图不足退出码 2", code == 2, str(code))
        check("正常图不足给出 DATA_TOO_FEW",
              any(event.get("code") == "DATA_TOO_FEW" for event in events), text[-300:])
        check("正常图不足时没有建输出目录",
              not (LAUNCHER_DIR / ".project_should_not_exist").exists())

        # 未知模式由 argparse 拦下（stderr 的 usage 提示不该污染测试输出）
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stderr(buffer):
                launcher.build_parser().parse_args(["--mode", "train"])
            check("未知模式要报错", False)
        except SystemExit:
            check("未知模式要报错", "invalid choice" in buffer.getvalue(), buffer.getvalue()[-200:])

        # ---- 11. 参数解析 ----------------------------------------------------
        args = launcher.build_parser().parse_args(
            ["--mode", "build-export", "--shots", "-1", "--rotation", "true",
             "--seed", "2", "--input-size", "448"]
        )
        check("shots 支持 -1（全部图片）", args.shots == -1, str(args.shots))
        check("rotation 是布尔字符串", launcher.parse_bool(args.rotation) is True)
        check("seed 透传", args.seed == 2)
        check("默认输入尺寸 448", launcher.build_parser().parse_args(
            ["--mode", "warm"]).input_size == 448)

        # ---- 12. 有真实权重时补一条主干断言 ---------------------------------
        try:
            checkpoint = core.resolve_checkpoint()
        except FileNotFoundError:
            checkpoint = None
        if checkpoint is not None:
            model = core.create_backbone(checkpoint, 224, "cpu")
            tokens = core.extract_patch_tokens(model, torch.zeros(1, 3, 224, 224))
            check("真实主干输出 [1, 16*16, 384]", tuple(tokens.shape) == (1, 256, 384),
                  str(tuple(tokens.shape)))
            check("真实主干 token 已 L2 归一前的范数合理",
                  float(tokens.norm(dim=-1).mean()) > 0.0)
        else:
            print("[SKIP] 本机没有 DINOv2 权重，跳过后端主干断言")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  !! {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
