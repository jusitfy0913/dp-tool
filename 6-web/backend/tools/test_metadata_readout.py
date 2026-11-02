# -*- coding: utf-8 -*-
"""metadata 推荐值回显的验收脚本（不需要 GPU、不需要启动 Web 服务）。

背景：勾选「沿用 metadata 默认阈值」时两个阈值输入框是只读的，值来自
`metadata.json`。在 v1.21 之前，界面只把输入框置灰，框里留着默认 0.5，
用户根本不知道文件里写的是什么，只能先跑一次推理再回头看结果行。
这次补了「只读文件 + 回显」这条链路，脚本覆盖它的五类风险：

1. 解析：阈值 / 模型类型 / 输入尺寸 / 校准精度都要能读出来；
2. 容错：缺字段、BOM、坏 JSON、布尔值冒充数字都不能抛异常；
3. 零值：`0.0` 是合法阈值（校准过的模型常见），不能被当成"没读到"；
4. 接口：正常返回 200、非白名单 403、文件不存在 400、坏 JSON 400；
5. 语义：该接口**不建引擎句柄**，只读文件（否则每改一次路径就加载一次模型）。

用法：
    D:\\Anaconda3\\envs\\dpenv\\python.exe 6-web\\backend\\tools\\test_metadata_readout.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import infer  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


def close(actual, expected, tol: float = 1e-9) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def write_metadata(path: Path, payload: dict | str, *, bom: bool = False) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    data = text.encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" + data) if bom else data)


# ---------------------------------------------------------------------------
# 1. 解析：字段齐全的 metadata
# ---------------------------------------------------------------------------
def test_full_metadata(tmp: Path) -> None:
    path = tmp / "metadata_full.json"
    write_metadata(path, {
        "model_type": "Patchcore",
        "input_size": [256, 256],
        "normalization": "none",
        "image_threshold": 0.17198532521724702,
        "pixel_threshold": 0.16340724378824234,
        "pred_scores": {"min": 0.0, "max": 1.0},
        "anomaly_maps": {"min": 0.0, "max": 1.0},
        "calibration": {"sample_accuracy": 1.0, "normal_image_count": 64},
    })

    info = infer.read_metadata_thresholds(str(path))
    check("解析：没有错误", info["error"] == "", info["error"])
    check("解析：图像阈值", close(info["image_threshold"], 0.17198532521724702),
          str(info["image_threshold"]))
    check("解析：像素阈值", close(info["pixel_threshold"], 0.16340724378824234),
          str(info["pixel_threshold"]))
    check("解析：模型类型", info["model_type"] == "Patchcore", str(info["model_type"]))
    check("解析：输入尺寸", info["input_size"] == [256, 256], str(info["input_size"]))
    check("解析：归一化方式", info["normalization"] == "none", str(info["normalization"]))
    check("解析：分数区间", info["pred_scores"] == [0.0, 1.0], str(info["pred_scores"]))
    check("解析：异常图区间", info["anomaly_maps"] == [0.0, 1.0], str(info["anomaly_maps"]))
    check("解析：校准精度", close(info["sample_accuracy"], 1.0), str(info["sample_accuracy"]))


# ---------------------------------------------------------------------------
# 2. 容错
# ---------------------------------------------------------------------------
def test_tolerant(tmp: Path) -> None:
    # 老版本 anomalib 导出的文件只有两个阈值，缺 model_type / input_size
    bare = tmp / "metadata_bare.json"
    write_metadata(bare, {"image_threshold": 0.5, "pixel_threshold": 0.4})
    info = infer.read_metadata_thresholds(str(bare))
    check("容错：缺 model_type 不抛错", info["error"] == "" and info["model_type"] == "", str(info))
    check("容错：缺 input_size 为 None", info["input_size"] is None, str(info["input_size"]))
    check("容错：缺校准区块时精度为 None", info["sample_accuracy"] is None, str(info["sample_accuracy"]))

    # Windows 记事本存的文件常带 BOM，直接 json.loads 会炸
    bom = tmp / "metadata_bom.json"
    write_metadata(bom, {"image_threshold": 0.25, "pixel_threshold": 0.15}, bom=True)
    info = infer.read_metadata_thresholds(str(bom))
    check("容错：容忍 BOM", info["error"] == "" and close(info["image_threshold"], 0.25), str(info))

    # 坏 JSON：只回报 error，绝不能让推理页整块崩掉
    broken = tmp / "metadata_broken.json"
    broken.write_text("{ this is not json", encoding="utf-8")
    info = infer.read_metadata_thresholds(str(broken))
    check("容错：坏 JSON 只回报 error", bool(info["error"]) and info["image_threshold"] is None, str(info))

    # 文件不存在同理（这个接口只用于展示，不能反过来挡推理）
    info = infer.read_metadata_thresholds(str(tmp / "nope.json"))
    check("容错：文件不存在只回报 error", bool(info["error"]), str(info))

    # 类型不对的字段：字符串数字、null、布尔值都不应被当成有效阈值
    weird = tmp / "metadata_weird.json"
    write_metadata(weird, {
        "image_threshold": "0.5",
        "pixel_threshold": None,
        "input_size": ["256", 256],
        "model_type": 123,
        "calibration": {"sample_accuracy": True},
    })
    info = infer.read_metadata_thresholds(str(weird))
    check("容错：字符串阈值不当数字用", info["image_threshold"] is None, str(info["image_threshold"]))
    check("容错：null 阈值不当数字用", info["pixel_threshold"] is None, str(info["pixel_threshold"]))
    # 尺寸写成 "256" 也认：JSON 里数字被引号包住的情况确实存在，能转就转
    check("容错：字符串尺寸可转换", info["input_size"] == [256, 256], str(info["input_size"]))
    check("容错：非字符串 model_type 置空", info["model_type"] == "", str(info["model_type"]))
    check("容错：布尔不当精度", info["sample_accuracy"] is None, str(info["sample_accuracy"]))

    # 真的转不了的尺寸：置 None，不能抛异常
    bad_size = tmp / "metadata_bad_size.json"
    write_metadata(bad_size, {"input_size": ["宽", "高"], "image_threshold": 0.5})
    info = infer.read_metadata_thresholds(str(bad_size))
    check("容错：非法 input_size 置 None", info["input_size"] is None, str(info["input_size"]))

    # 尺寸长度不是 2 也置 None
    tri = tmp / "metadata_tri.json"
    write_metadata(tri, {"input_size": [256, 256, 3]})
    info = infer.read_metadata_thresholds(str(tri))
    check("容错：长度不为 2 的 input_size 置 None", info["input_size"] is None, str(info["input_size"]))


# ---------------------------------------------------------------------------
# 3. 零值
# ---------------------------------------------------------------------------
def test_zero_threshold(tmp: Path) -> None:
    # 校准过的模型会把阈值压到 0（正常样本分数恒为 0 时），这必须是"读到了 0"，
    # 而不是"没读到"——前端拿 0 和 null 的展示是完全不同的。
    path = tmp / "metadata_zero.json"
    write_metadata(path, {"image_threshold": 0.0, "pixel_threshold": 0.0})
    info = infer.read_metadata_thresholds(str(path))
    check("零值：0.0 被当作有效阈值", info["image_threshold"] == 0.0, str(info["image_threshold"]))
    check("零值：像素阈值同样", info["pixel_threshold"] == 0.0, str(info["pixel_threshold"]))


# ---------------------------------------------------------------------------
# 4. 接口层
# ---------------------------------------------------------------------------
def test_api(tmp: Path) -> None:
    from fastapi.testclient import TestClient

    import app as webapp

    prior_roots = os.environ.get("TRTFORGE_BROWSE_ROOTS", "")
    os.environ["TRTFORGE_BROWSE_ROOTS"] = str(tmp) + os.pathsep + str(BACKEND.parent)
    config.clear_runtime_allowed()
    try:
        client = TestClient(webapp.app)

        good = tmp / "api" / "metadata.json"
        good.parent.mkdir(parents=True, exist_ok=True)
        write_metadata(good, {
            "model_type": "Padim",
            "input_size": [512, 512],
            "image_threshold": 0.31,
            "pixel_threshold": 0.22,
            "calibration": {"sample_accuracy": 0.96},
        })

        response = client.get("/api/infer/anomalib/metadata", params={"path": str(good)})
        check("接口：正常读取 200", response.status_code == 200, f"{response.status_code} {response.text[:200]}")
        body = response.json()
        check("接口：下发图像阈值", close(body.get("image_threshold"), 0.31), str(body.get("image_threshold")))
        check("接口：下发像素阈值", close(body.get("pixel_threshold"), 0.22), str(body.get("pixel_threshold")))
        check("接口：下发模型类型", body.get("model_type") == "Padim", str(body.get("model_type")))
        check("接口：下发输入尺寸", body.get("input_size") == [512, 512], str(body.get("input_size")))
        check("接口：下发校准精度", close(body.get("sample_accuracy"), 0.96), str(body.get("sample_accuracy")))
        check("接口：回显实际读取的路径", body.get("path") == str(good), str(body.get("path")))

        # 空路径：不是 500，而是明确的 400，前端能把这句话直接显示出来
        response = client.get("/api/infer/anomalib/metadata", params={"path": ""})
        check("接口：空路径 400", response.status_code == 400, str(response.status_code))

        # 文件不存在
        response = client.get("/api/infer/anomalib/metadata", params={"path": str(tmp / "no_such.json")})
        check("接口：文件不存在 400", response.status_code == 400, str(response.status_code))

        # 坏 JSON：400 + 可读原因
        broken = tmp / "api" / "metadata_broken.json"
        broken.write_text("{ nope", encoding="utf-8")
        response = client.get("/api/infer/anomalib/metadata", params={"path": str(broken)})
        check("接口：坏 JSON 400", response.status_code == 400, str(response.status_code))
        check("接口：坏 JSON 带原因", "读取失败" in response.text, response.text[:200])

        # 白名单外：换一个不在 TRTFORGE_BROWSE_ROOTS 里的目录
        outside = Path(tempfile.gettempdir()).parent / "tf_meta_outside_xyz"
        outside.mkdir(exist_ok=True)
        stranger = outside / "metadata.json"
        write_metadata(stranger, {"image_threshold": 0.5})
        response = client.get("/api/infer/anomalib/metadata", params={"path": str(stranger)})
        check("接口：白名单外 403", response.status_code == 403, f"{response.status_code} {response.text[:200]}")

        # 语义：这个接口只读文件，不得创建引擎句柄
        original_lib = infer._anomalib_lib
        created: list[str] = []

        class RecordingLib:
            def ai_create(self, *args, **kwargs):
                created.append("created")
                raise AssertionError("回显接口不应该建引擎句柄")

        infer._anomalib_lib = RecordingLib()
        try:
            client.get("/api/infer/anomalib/metadata", params={"path": str(good)})
        finally:
            infer._anomalib_lib = original_lib
        check("接口：不加载 DLL / 不建引擎句柄", not created, str(created))
    finally:
        if prior_roots:
            os.environ["TRTFORGE_BROWSE_ROOTS"] = prior_roots
        else:
            os.environ.pop("TRTFORGE_BROWSE_ROOTS", None)
        config.clear_runtime_allowed()


# ---------------------------------------------------------------------------
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tf_meta_test_"))
    try:
        test_full_metadata(tmp)
        test_tolerant(tmp)
        test_zero_threshold(tmp)
        test_api(tmp)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
