# -*- coding: utf-8 -*-
"""数据集域路由：扫描、体检、划分导出、指纹。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
import dataset
import datasetqc
import fingerprint

# 共享状态：路径白名单（见 web_state.py）
from web_state import _ensure_allowed_path

router = APIRouter()
# ---------------------------------------------------------------------------
# 数据集工作台：扫描 / 划分预览 / 导出
# ---------------------------------------------------------------------------
class DatasetScanRequest(BaseModel):
    """扫描请求。ratios 只影响预览，不落盘。"""

    source: str = Field(..., description="YOLO 数据集目录、data.yaml 或 .tfanno 文件")
    task: str = Field(default="auto", description="auto / detect / segment / pose / obb")
    train: int = Field(default=80, ge=0, le=100)
    val: int = Field(default=20, ge=0, le=100)
    group_by: str = Field(
        default="none",
        description="划分口径：none 按图片 / prefix 按文件名前缀 / dir 按目录（同组不跨划分）",
    )
    limit: int = Field(default=200, ge=1, le=2000, description="预览列表最多返回多少张")
    ok_dirs: list[str] = Field(
        default_factory=list,
        description="良品目录（相对数据源根）：目录里的图按空标签处理，默认只进验收划分",
    )
    ok_split: str = Field(default="val", description="良品图进哪个划分：val / test")


class DatasetExportRequest(DatasetScanRequest):
    """导出请求。额外带上格式、输出目录与覆盖开关。"""

    format: str = Field(default="yolo", description="yolo / coco / voc")
    output: str = Field(..., description="导出目标目录")
    overwrite: bool = Field(default=False, description="目标目录非空时是否允许覆盖")
    with_qc: bool = Field(default=False, description="顺带体检，把结论写进 manifest（不影响导出结果）")


class DatasetQcRequest(BaseModel):
    """体检请求。只读，不落盘。"""

    source: str = Field(..., description="YOLO 数据集目录、data.yaml 或 .tfanno 文件")
    task: str = Field(default="auto", description="auto / detect / segment / pose / obb / classify")
    threshold: int = Field(default=8, ge=0, le=256, description="256 位 dHash 的汉明距离上限")
    limit: int = Field(default=500, ge=1, le=5000, description="问题列表最多返回多少条")
    ok_dirs: list[str] = Field(
        default_factory=list, description="良品目录：这些目录里的图不报 LBL_MISSING"
    )


def _ensure_dataset_path(path: str) -> None:
    """数据集路径沿用目录浏览的白名单，避免接口被用来读任意磁盘。"""
    _ensure_allowed_path(path)


@router.post("/api/dataset/scan")
def dataset_scan(request: DatasetScanRequest) -> dict:
    """扫描数据集并返回按比例划分后的预览，不写任何文件。"""
    _ensure_dataset_path(request.source)
    try:
        project = dataset.load_project(
            request.source, request.task, ok_dirs=request.ok_dirs, ok_split=request.ok_split
        )
        return dataset.preview(
            project,
            request.train,
            request.val,
            request.limit,
            group_by=request.group_by,
            ok_split=request.ok_split,
        )
    except dataset.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"读取数据集失败：{exc}") from exc


@router.post("/api/dataset/qc")
def dataset_qc(request: DatasetQcRequest) -> dict:
    """数据集体检：同底板分组、跨划分泄漏、标注问题。只读，不写任何数据文件。

    为什么和「指纹」分开：指纹回答「是不是同一份数据」，体检回答「这份数据本身
    干不干净」。两者都只读，但体检要解码图片（秒级），不适合塞进导出/训练的必经路径。
    """
    _ensure_dataset_path(request.source)
    try:
        return datasetqc.inspect(
            request.source,
            request.task,
            threshold=request.threshold,
            limit=request.limit,
            ok_dirs=request.ok_dirs,
        )
    except dataset.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"体检失败：{exc}") from exc


@router.post("/api/dataset/export")
def dataset_export(request: DatasetExportRequest) -> dict:
    """扫描 → 划分 → 导出，一次请求完成，不需要前端保存划分状态。"""
    _ensure_dataset_path(request.source)
    _ensure_dataset_path(request.output)
    try:
        return dataset.run_export(
            source=request.source,
            task=request.task,
            train=request.train,
            val=request.val,
            fmt=request.format,
            output=request.output,
            overwrite=request.overwrite,
            group_by=request.group_by,
            with_qc=request.with_qc,
            ok_dirs=request.ok_dirs,
            ok_split=request.ok_split,
        )
    except dataset.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"导出失败：{exc}") from exc


@router.get("/api/dataset/fingerprint")
def dataset_fingerprint(source: str) -> dict:
    """只读算出某个 ``dataset.yaml`` 的数据集指纹，不写任何文件。

    用途：页面在用户选完数据集、还没开训时就能显示「本次将使用哪批数据」；
    以及排查「两次实验为什么不可比」时手动比对。**不缓存**：源数据随时会变，
    读到旧指纹比读不到更糟。
    """
    _ensure_dataset_path(source)
    report = fingerprint.compute(source)
    report["description"] = fingerprint.describe(report)
    report["version"] = config.APP_VERSION
    return report
