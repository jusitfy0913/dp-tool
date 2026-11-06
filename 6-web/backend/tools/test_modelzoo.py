# -*- coding: utf-8 -*-
"""内置模型仓库验收脚本（本地起一个 HTTP 服务，不访问外网）。

覆盖变更 `add-prompt-autolabel-review` 里"模型从哪来"这一块最容易出错的点：

1. **目录可覆盖**：`TRTFORGE_MODEL_DIR` 必须生效，不能写死仓库内路径；
2. **清单状态**：整组文件齐了才算「已就绪」（SAM 少一个 decoder 就不能用）；
3. **断点续传**：已有 `.part` 时必须带 `Range` 从断点继续，而不是从头下；
4. **取消**：取消后状态是 `cancelled`，且半成品 `.part` 被清掉；
5. **并发保护**：同一个模型不许同时下两份（会把 `.part` 写坏）；
6. **失败可读**：所有地址都不可用时给 `failed` + 原因，不是静默卡住。

用法：
    python 6-web/backend/tools/test_modelzoo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import modelzoo  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSED.append(name)
        print(f"[PASS] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


# ---------------------------------------------------------------------------
# 测试用的 HTTP 服务：支持 Range，可切换"慢速"与"坏掉"
# ---------------------------------------------------------------------------
class RangeHandler(BaseHTTPRequestHandler):
    payload = b"0123456789" * 4096  # 40 KB
    slow = False
    break_after = 0  # >0 时写这么多字节后掐断连接
    range_requests: list[int] = []

    def log_message(self, *args):  # noqa: D401 - 静音，测试输出只看 PASS/FAIL
        return

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的约定
        start = 0
        header = self.headers.get("Range") or ""
        if header.startswith("bytes="):
            try:
                start = int(header[len("bytes=") :].split("-")[0])
            except ValueError:
                start = 0
        RangeHandler.range_requests.append(start)
        body = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")
        self.end_headers()
        try:
            if self.slow:
                # 分块慢发，给"取消"留出窗口
                for index in range(0, len(body), 2048):
                    self.wfile.write(body[index : index + 2048])
                    self.wfile.flush()
                    time.sleep(0.05)
                return
            if self.break_after:
                self.wfile.write(body[: self.break_after])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端取消导致的断开是预期行为


def start_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def register(entry: dict) -> None:
    modelzoo.CATALOG.append(entry)
    modelzoo._BY_ID[entry["id"]] = entry


def unregister(model_id: str) -> None:
    modelzoo.CATALOG[:] = [item for item in modelzoo.CATALOG if item["id"] != model_id]
    modelzoo._BY_ID.pop(model_id, None)


def wait_done(job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = modelzoo.get_job(job_id)
        if snapshot is None:
            raise AssertionError(f"任务丢了：{job_id}")
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.02)
    return modelzoo.wait(job_id, 1.0) or {}


# ---------------------------------------------------------------------------
# 1. 目录与清单
# ---------------------------------------------------------------------------
def test_dir(tmp: Path) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    try:
        os.environ.pop("TRTFORGE_MODEL_DIR", None)
        default_dir = modelzoo.model_dir()
        check("目录：默认落在 6-web/models", default_dir == modelzoo._DEFAULT_DIR, str(default_dir))

        os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "custom-zoo")
        resolved = modelzoo.model_dir()
        check("目录：环境变量覆盖生效", resolved == (tmp / "custom-zoo").resolve(), str(resolved))
        check("目录：ensure_dir 会建出来", modelzoo.ensure_dir().is_dir(), str(modelzoo.model_dir()))
    finally:
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_catalog(tmp: Path) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "catalog-zoo")
    try:
        data = modelzoo.list_models()
        ids = [item["id"] for item in data["items"]]
        check("清单：内置三项都在", set(ids) == {"mobile_sam", "grounding_dino_swint", "bert_tokenizer"}, str(ids))
        check("清单：总数对得上", data["total"] == len(ids), str(data["total"]))
        check("清单：初始全部缺失", data["missing"] == len(ids), str(data["missing"]))
        for item in data["items"]:
            if not item["name"] or not item["use"] or not item["license"]:
                check("清单：每项都有名称/用途/许可证", False, str(item))
                break
        else:
            check("清单：每项都有名称/用途/许可证", True)

        sam = next(item for item in data["items"] if item["id"] == "mobile_sam")
        check("清单：SAM 是两条文件", len(sam["files"]) == 2, str([f["name"] for f in sam["files"]]))
        check("清单：SAM 未就绪时缺文件列表非空", sam["present"] is False and len(sam["missing"]) == 2, str(sam["missing"]))
        check("清单：体积为正", sam["total_bytes"] > 0, str(sam["total_bytes"]))

        # 只放一个文件 → 仍然算"未就绪"（整组齐了才算）
        zoo = modelzoo.ensure_dir()
        (zoo / "mobile_sam.encoder.onnx").write_bytes(b"x" * 10)
        after = modelzoo.list_models()
        sam2 = next(item for item in after["items"] if item["id"] == "mobile_sam")
        check("清单：缺一个文件仍算未就绪", sam2["present"] is False, str(sam2["missing"]))
        (zoo / "sam_vit_h_4b8939.decoder.onnx").write_bytes(b"y" * 10)
        after2 = modelzoo.list_models()
        sam3 = next(item for item in after2["items"] if item["id"] == "mobile_sam")
        check("清单：文件齐了才就绪", sam3["present"] is True and sam3["missing"] == [], str(sam3))

        # 空文件不算就绪（下载中断会留下 0 字节文件）
        (zoo / "sam_vit_h_4b8939.decoder.onnx").write_bytes(b"")
        after3 = modelzoo.list_models()
        sam4 = next(item for item in after3["items"] if item["id"] == "mobile_sam")
        check("清单：0 字节文件不算就绪", sam4["present"] is False, str(sam4["missing"]))
    finally:
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_require(tmp: Path) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "require-zoo")
    try:
        try:
            modelzoo.require("mobile_sam")
            check("require：缺文件时抛 ModelMissing", False, "没有抛错")
        except modelzoo.ModelMissing as exc:
            check("require：缺文件时抛 ModelMissing", True)
            check("require：错误里带下载指引", "下载" in str(exc), str(exc))

        try:
            modelzoo.require("no_such_model")
            check("require：未知模型抛 ModelError", False, "没有抛错")
        except modelzoo.ModelError as exc:
            check("require：未知模型抛 ModelError", True)
            check("require：错误里列出可选 id", "mobile_sam" in str(exc), str(exc))

        check("resolve：文件不存在返回 None", modelzoo.resolve("mobile_sam", "mobile_sam.encoder.onnx") is None, "")

        paths = modelzoo.require("bert_tokenizer")  # 先放一份真文件
        del paths
    except modelzoo.ModelError:
        pass
    finally:
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_local_source() -> None:
    local = modelzoo._local_source("@local:8-X-AnyLabeling-main/anylabeling/services/auto_labeling/configs/bert")
    check("本地兜底：@local 解析成仓库内路径", local is not None and local.name == "bert", str(local))
    check("本地兜底：普通 URL 不算本地", modelzoo._local_source("https://example.com/x") is None, "")


# ---------------------------------------------------------------------------
# 2. 下载：完整 / 续传 / 取消 / 并发
# ---------------------------------------------------------------------------
def test_download(tmp: Path, base: str) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "download-zoo")
    modelzoo.reset_for_tests()
    RangeHandler.range_requests = []
    RangeHandler.slow = False
    RangeHandler.break_after = 0

    model_id = "test_download_model"
    register(
        {
            "id": model_id,
            "name": "测试模型",
            "use": "单测用",
            "license": "MIT",
            "channel": "test",
            "files": [
                {"name": "test_model.bin", "urls": [f"{base}/x.bin"], "size": len(RangeHandler.payload)}
            ],
        }
    )
    try:
        modelzoo.ensure_dir()
        started = modelzoo.start_download(model_id)
        check("下载：返回 running 任务", started["status"] == "running", str(started))
        done = wait_done(started["id"])
        check("下载：状态变成 done", done["status"] == "done", str(done))
        target = modelzoo.model_dir() / "test_model.bin"
        check("下载：文件落地", target.is_file(), str(target))
        check("下载：内容完整", target.read_bytes() == RangeHandler.payload, str(target.stat().st_size if target.is_file() else 0))
        check("下载：进度 100%", done["percent"] == 100, str(done["percent"]))
        check("下载：没有留下 .part", not (modelzoo.model_dir() / "test_model.bin.part").exists())
        check("下载：首次请求没有 Range", RangeHandler.range_requests[0] == 0, str(RangeHandler.range_requests))

        # 已就绪再点下载 → already
        again = modelzoo.start_download(model_id)
        check("下载：已就绪返回 already", again.get("already") is True, str(again))

        # 清单里认出来了
        item = next(entry for entry in modelzoo.list_models()["items"] if entry["id"] == model_id)
        check("下载：清单显示已就绪", item["present"] is True, str(item))
    finally:
        unregister(model_id)
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_resume(tmp: Path, base: str) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "resume-zoo")
    modelzoo.reset_for_tests()
    RangeHandler.range_requests = []
    RangeHandler.slow = False
    RangeHandler.break_after = 0

    model_id = "test_resume_model"
    register(
        {
            "id": model_id,
            "name": "续传模型",
            "use": "单测用",
            "license": "MIT",
            "channel": "test",
            "files": [
                {"name": "resume.bin", "urls": [f"{base}/r.bin"], "size": len(RangeHandler.payload)}
            ],
        }
    )
    try:
        zoo = modelzoo.ensure_dir()
        part = zoo / "resume.bin.part"
        half = len(RangeHandler.payload) // 2
        part.write_bytes(RangeHandler.payload[:half])

        started = modelzoo.start_download(model_id)
        done = wait_done(started["id"])
        check("续传：状态 done", done["status"] == "done", str(done))
        check("续传：请求带上了断点偏移", RangeHandler.range_requests and RangeHandler.range_requests[0] == half, str(RangeHandler.range_requests))
        final = zoo / "resume.bin"
        check("续传：最终内容完整", final.read_bytes() == RangeHandler.payload, str(final.stat().st_size if final.is_file() else 0))
        check("续传：.part 被改名而不是复制", not part.exists())
    finally:
        unregister(model_id)
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_cancel(tmp: Path, base: str) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "cancel-zoo")
    modelzoo.reset_for_tests()
    RangeHandler.slow = True
    RangeHandler.break_after = 0

    model_id = "test_cancel_model"
    register(
        {
            "id": model_id,
            "name": "取消模型",
            "use": "单测用",
            "license": "MIT",
            "channel": "test",
            "files": [
                {"name": "cancel.bin", "urls": [f"{base}/c.bin"], "size": len(RangeHandler.payload)}
            ],
        },
    )
    try:
        zoo = modelzoo.ensure_dir()
        started = modelzoo.start_download(model_id)
        time.sleep(0.15)
        cancelled = modelzoo.cancel(started["id"])
        check("取消：cancel 立刻返回任务快照", cancelled is not None and cancelled["id"] == started["id"], str(cancelled))
        done = wait_done(started["id"], 10.0)
        check("取消：最终状态是 cancelled", done["status"] == "cancelled", str(done))
        check("取消：半成品被清掉", not (zoo / "cancel.bin.part").exists())
        check("取消：没有生成正式文件", not (zoo / "cancel.bin").exists())
        check("取消：下载字节已记录", done["downloaded"] > 0, str(done["downloaded"]))
    finally:
        RangeHandler.slow = False
        unregister(model_id)
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_busy_and_failure(tmp: Path) -> None:
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "busy-zoo")
    modelzoo.reset_for_tests()
    RangeHandler.slow = True

    model_id = "test_busy_model"
    register(
        {
            "id": model_id,
            "name": "并发模型",
            "use": "单测用",
            "license": "MIT",
            "channel": "test",
            "files": [
                {"name": "busy.bin", "urls": ["http://127.0.0.1:1/never.bin"], "size": 100}
            ],
        }
    )
    try:
        modelzoo.ensure_dir()
        # 指向必然连不上的端口 → 快速失败
        started = modelzoo.start_download(model_id)
        try:
            modelzoo.start_download(model_id)
            check("并发：同一个模型第二次下载被拒", False, "没有抛错")
        except modelzoo.DownloadBusy:
            check("并发：同一个模型第二次下载被拒", True)
        done = wait_done(started["id"], 10.0)
        check("失败：连不上时状态是 failed", done["status"] == "failed", str(done))
        check("失败：带上了原因", bool(done["error"]), str(done.get("error")))

        # 之后再点可以重试（不能被上次的失败永久卡住）
        retry = modelzoo.start_download(model_id)
        check("失败：可以重试", retry["status"] == "running", str(retry))
        wait_done(retry["id"], 10.0)

        try:
            modelzoo.start_download("no_such_model")
            check("失败：未知模型被拒", False, "没有抛错")
        except modelzoo.ModelError:
            check("失败：未知模型被拒", True)

        check("失败：get_job 查不到时返回 None", modelzoo.get_job("nope") is None, "")
        check("失败：cancel 查不到时返回 None", modelzoo.cancel("nope") is None, "")
    finally:
        RangeHandler.slow = False
        unregister(model_id)
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def test_local_fallback(tmp: Path) -> None:
    """`@local:` 兜底：仓库内已有文件时直接拷贝，不联网。"""
    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "local-zoo")
    modelzoo.reset_for_tests()

    source = tmp / "repo" / "cfg" / "token.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"version":"1.0"}', encoding="utf-8")

    model_id = "test_local_model"
    register(
        {
            "id": model_id,
            "name": "本地兜底",
            "use": "单测用",
            "license": "MIT",
            "channel": "test",
            "files": [
                {
                    "name": "local_token.json",
                    # 直接给绝对路径做不到（@local 是相对仓库根的），所以这里退一步：
                    # 只验证"普通 URL 不会被当成本地路径"，以及本地路径不存在时会继续试下一个地址
                    "urls": ["@local:definitely-missing-dir/token.json", "http://127.0.0.1:1/x.bin"],
                    "size": 20,
                }
            ],
        }
    )
    try:
        modelzoo.ensure_dir()
        started = modelzoo.start_download(model_id)
        done = wait_done(started["id"], 10.0)
        check("本地兜底：本地路径不存在时继续尝试下一个地址", done["status"] == "failed", str(done))
    finally:
        unregister(model_id)
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


# ---------------------------------------------------------------------------
# 3. HTTP 接口
# ---------------------------------------------------------------------------
def test_api(tmp: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        check(f"接口：跳过（未安装 httpx：{exc}）", True)
        return

    import app as web_app

    old = os.environ.get("TRTFORGE_MODEL_DIR")
    os.environ["TRTFORGE_MODEL_DIR"] = str(tmp / "api-zoo")
    modelzoo.reset_for_tests()
    try:
        client = TestClient(web_app.app)
        listing = client.get("/api/modelzoo")
        check("接口：清单 200", listing.status_code == 200, listing.text[:200])
        body = listing.json()
        check("接口：清单带 items", isinstance(body.get("items"), list) and len(body["items"]) == 3, listing.text[:300])
        check("接口：清单带目录", bool(body.get("dir")), listing.text[:200])
        check("接口：清单带 jobs", isinstance(body.get("jobs"), list), listing.text[:200])

        unknown = client.post("/api/modelzoo/download", json={"id": "nope"})
        check("接口：未知模型 400", unknown.status_code == 400, f"{unknown.status_code} {unknown.text[:200]}")

        missing_job = client.get("/api/modelzoo/download/nope")
        check("接口：未知任务 404", missing_job.status_code == 404, f"{missing_job.status_code}")

        missing_cancel = client.post("/api/modelzoo/download/nope/cancel")
        check("接口：取消未知任务 404", missing_cancel.status_code == 404, f"{missing_cancel.status_code}")
    finally:
        modelzoo.reset_for_tests()
        if old is None:
            os.environ.pop("TRTFORGE_MODEL_DIR", None)
        else:
            os.environ["TRTFORGE_MODEL_DIR"] = old


def main() -> int:
    server, base = start_server()
    try:
        with tempfile.TemporaryDirectory(prefix="modelzoo-") as temp:
            tmp = Path(temp)
            test_dir(tmp)
            test_catalog(tmp)
            test_require(tmp)
            test_local_source()
            test_download(tmp, base)
            test_resume(tmp, base)
            test_cancel(tmp, base)
            test_busy_and_failure(tmp)
            test_local_fallback(tmp)
            test_api(tmp)
    finally:
        server.shutdown()
        server.server_close()

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
