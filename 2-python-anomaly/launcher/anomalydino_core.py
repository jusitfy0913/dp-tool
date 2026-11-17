"""AnomalyDINO 的核心算法实现：DINOv2 主干 + 冻结特征记忆库。

为什么单独一个模块：
1. 这套算法**不依赖 anomalib / lightning**，只是「主干前向 + 最近邻距离」。
   把它和 anomalib 的启动器混在一个文件里，会让 import 关系变得难以判断，
   也会让使用人员误以为跑 AnomalyDINO 需要装 Lightning。
2. 所有纯逻辑（位置编码插值、建库、分块相似度、top-1% 聚合）都放在这里，
   可以被 `6-web/backend/tools/` 下的自测脚本直接 import，不需要起子进程。

关键事实（已在本机 dpenv / timm 1.0.14 / torch 2.3.0 上实测）：

- 直接用 timm 的 `vit_small_patch14_dinov2.lvd142m` 加载官方 DINOv2 权重，
  与上游 `torch.hub` 模型**并不等价**（逐 token 余弦最低只有 0.9928）。
  原因是上游 `DinoVisionTransformer` 的 `interpolate_offset = 0.1`，
  它把位置编码按 `scale_factor=(grid+0.1)/M` 做 bicubic 插值，而不是 `size=(grid, grid)`。
- 按上游公式预插值 `pos_embed` 后，两者输出**逐元素完全相同**（最大差 0.0）。
  所以本模块固定走「预插值 + 静态 img_size」这条路，运行期不需要 torch.hub。
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# DINOv2 官方权重的原始位置编码网格：518 / 14 = 37。
# 这是一个与权重绑定的事实，不是可调参数：官方 ckpt 的 pos_embed 就是 1 + 37*37 个。
DINOV2_BASE_GRID = 37

# 上游 DinoVisionTransformer 的默认插值偏移。改它就会和论文实现不一致。
DINOV2_INTERPOLATE_OFFSET = 0.1

# timm 里的等价模型名。注意版本要求很低（0.9 起就有），当前 1.0.14 完全够用。
TIMM_DINOV2_SMALL = "vit_small_patch14_dinov2.lvd142m"

# ViT-S/14 的 patch 尺寸与特征维度。
PATCH_SIZE = 14
EMBED_DIM = 384

# 分块算相似度：每块 4096 行库，1024×4096×4B ≈ 16.8 MB 中间张量。
# 不这么做的话，full-shot（225 张 = 230400 行）会直接爆显存。
DEFAULT_BANK_CHUNK = 4096

# 上游 `mean_top1p` 的比例：全网格距离降序前 1% 取均值。
TOP_RATIO = 0.01

#: 官方权重文件名。所有候选路径都由它拼出来，改名只改这里。
CHECKPOINT_NAME = "dinov2_vits14_pretrain.pth"


def checkpoint_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """算权重的 sha256，用于写进 metadata 的溯源字段。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_pos_embed(pos_embed: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """把官方 37×37 的位置编码插值到目标网格，公式与上游完全一致。

    `pos_embed` 形状为 `[1, 1 + 37*37, dim]`，第 0 个是 cls token 的位置编码。
    返回 `[1, 1 + grid_h*grid_w, dim]`。

    这里**必须**用 `scale_factor=(grid+0.1)/37`，不能用 `size=(grid, grid)`：
    两种 bicubic 采样网格不同，实测逐 token 余弦会从 1.0 掉到 0.991。
    """
    if pos_embed.dim() != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"pos_embed 形状应为 [1, N, dim]，实际 {tuple(pos_embed.shape)}")
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"网格尺寸必须为正，实际 {grid_h}x{grid_w}")

    pos_embed = pos_embed.float()
    patches = pos_embed.shape[1] - 1
    base_grid = int(round(math.sqrt(patches)))
    if base_grid * base_grid != patches:
        raise ValueError(f"pos_embed 的 patch 数 {patches} 不是完全平方数，无法还原网格")

    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:].reshape(1, base_grid, base_grid, -1).permute(0, 3, 1, 2)
    scale_y = (grid_h + DINOV2_INTERPOLATE_OFFSET) / base_grid
    scale_x = (grid_w + DINOV2_INTERPOLATE_OFFSET) / base_grid
    patch_pos = F.interpolate(
        patch_pos,
        mode="bicubic",
        antialias=False,
        scale_factor=(scale_y, scale_x),
    )
    # 浮点 scale_factor 偶尔会多算一格，这里按目标网格裁一下，避免 pos_embed 长度不匹配。
    patch_pos = patch_pos[:, :, :grid_h, :grid_w]
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
    return torch.cat((cls_pos, patch_pos), dim=1)


def grid_for_resolution(resolution: int) -> tuple[int, int]:
    """把边长换算成网格。分辨率必须是 patch 的整数倍，否则网格不整齐。"""
    if resolution <= 0 or resolution % PATCH_SIZE != 0:
        raise ValueError(
            f"分辨率必须是 {PATCH_SIZE} 的倍数（例如 448），实际 {resolution}"
        )
    side = resolution // PATCH_SIZE
    return side, side


def default_checkpoint_candidates(explicit: str | os.PathLike[str] | None = None) -> list[Path]:
    """按「显式指定 → 仓库 modelzoo → 启动器自带目录 → torch.hub 缓存」的顺序列出候选权重。

    顺序的依据是「谁更可能是有意放的」：
    - `--checkpoint` 与 modelzoo 都是人主动准备的，优先；
    - `pre_trained/` 是开发者手工放的位置；
    - hub 缓存是 `torch.hub.load` 留下的副产物，最后才用。
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    # modelzoo 下载的落点：安装形态下由启动器把 TRTFORGE_MODEL_DIR 指到 <安装目录>\models
    zoo_dir = os.environ.get("TRTFORGE_MODEL_DIR", "").strip()
    if zoo_dir:
        candidates.append(Path(zoo_dir) / "dinov2" / CHECKPOINT_NAME)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "6-web" / "models" / "dinov2" / CHECKPOINT_NAME)
    repo_models = Path(__file__).resolve().parent / "pre_trained" / CHECKPOINT_NAME
    candidates.append(repo_models)
    hub_cache = Path(os.path.expanduser("~")) / ".cache" / "torch" / "hub" / "checkpoints"
    candidates.append(hub_cache / CHECKPOINT_NAME)
    return candidates


def resolve_checkpoint(explicit: str | os.PathLike[str] | None = None) -> Path:
    """找到可用的 DINOv2 权重，找不到时给出可执行的建议。"""
    tried = default_checkpoint_candidates(explicit)
    for candidate in tried:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    listed = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(
        "找不到 DINOv2 权重 dinov2_vits14_pretrain.pth，已尝试：\n"
        f"{listed}\n"
        "请在网页「模型仓库」里下载 DINOv2（会落到 models/dinov2/ 下），"
        "或把它放到 2-python-anomaly/launcher/pre_trained/ 下，或用 --checkpoint 显式指定。"
    )


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """读权重。用 weights_only 兼容未来 torch 版本的默认值变化。"""
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 没有 weights_only 参数
        state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"权重文件不是 state_dict：{checkpoint}")
    # 有些导出会把权重包在 "state_dict" 或 "model" 键里，这里顺手兼容一下。
    for key in ("state_dict", "model"):
        inner = state.get(key)
        if isinstance(inner, dict) and any(k.endswith("pos_embed") or k == "pos_embed" for k in inner):
            return dict(inner)
    return dict(state)


def create_backbone(
    checkpoint: Path,
    resolution: int,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """创建与上游等价的 DINOv2 ViT-S/14 主干。

    返回的模型 `forward_features(x)[0, 1:]` 就是 patch token（去掉 cls）。
    """
    import timm  # 延迟 import：check-env 阶段需要先判断 timm 是否存在

    grid_h, grid_w = grid_for_resolution(resolution)
    state = load_state_dict(checkpoint)
    if "pos_embed" not in state:
        raise KeyError(f"权重里没有 pos_embed，可能不是 DINOv2 官方 ckpt：{checkpoint}")
    state["pos_embed"] = build_pos_embed(state["pos_embed"], grid_h, grid_w)

    model = timm.create_model(
        TIMM_DINOV2_SMALL,
        pretrained=False,
        num_classes=0,
        img_size=(resolution, resolution),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # 唯一允许缺失/多余的键是 mask_token：它只服务 MAE 式预训练，推理路径用不到。
    bad_missing = [k for k in missing if "mask_token" not in k]
    bad_unexpected = [k for k in unexpected if "mask_token" not in k]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"权重与 {TIMM_DINOV2_SMALL} 结构不匹配："
            f"缺少 {bad_missing[:5]}，多余 {bad_unexpected[:5]}"
        )
    model.eval()
    return model.to(device)


@torch.inference_mode()
def extract_patch_tokens(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """取 patch token，形状 `[N, grid_h*grid_w, dim]`。"""
    features = model.forward_features(batch)
    if features.dim() != 3:
        raise ValueError(f"主干输出应为 [N, T, C]，实际 {tuple(features.shape)}")
    return features[:, 1:, :]


def build_bank(token_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """把多张图的 patch token 拼成记忆库，并做 L2 归一化。

    行数 = 图数 × 网格单元数。上游 `get_model` 里对参考图也做了 L2 归一化
    （faiss 的 `normalize_L2`），所以归一化放在这里，图内只做点积。
    """
    if not token_tensors:
        raise ValueError("建库至少需要一张正常图的特征")
    flat = [t.reshape(-1, t.shape[-1]).float() for t in token_tensors]
    bank = torch.cat(flat, dim=0)
    if bank.shape[0] == 0:
        raise ValueError("记忆库为空，检查正常图像是否可读")
    return F.normalize(bank, dim=-1)


def bank_digest(bank: torch.Tensor) -> str:
    """库内容的指纹：同样的图、同样的顺序、同样的分辨率 → 同样的指纹。"""
    contiguous = bank.detach().to(torch.float32).contiguous().cpu().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def top1p_count(cells: int, ratio: float = TOP_RATIO) -> int:
    """上游 `mean_top1p` 的取整方式：`int(len * 0.01)`，最小为 1。"""
    return max(1, int(cells * ratio))


class BakedAnomalyDINO(nn.Module):
    """把记忆库固化成常量的推理图，直接输出 `anomaly_map` 与 `pred_score`。

    输出契约（与 `anomalib_infer.dll` 的挑选规则严格对齐）：
    - `anomaly_map`：`[1, 1, grid_h, grid_w]`，**必须是元素数最多的输出**
    - `pred_score`：`[1, 1]`，只有一个元素

    DLL 的选择逻辑里有一条「元素数比当前异常图大的输出会顶替它」
    （`anomalib_infer_c.cpp:464`），所以这里只允许导出这两个张量。
    """

    def __init__(
        self,
        backbone: nn.Module,
        bank: torch.Tensor,
        grid_h: int,
        grid_w: int,
        chunk_rows: int = DEFAULT_BANK_CHUNK,
        top_ratio: float = TOP_RATIO,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.chunk_rows = int(chunk_rows)
        self.top_k = top1p_count(self.grid_h * self.grid_w, top_ratio)
        # 库作为 buffer 会进 state_dict，导出时被固化成 ONNX 常量（Initializer）。
        self.register_buffer("bank", bank.float().contiguous(), persistent=True)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = extract_patch_tokens(self.backbone, image)[0]  # [cells, dim]
        tokens = F.normalize(tokens.float(), dim=-1)

        rows = self.bank.shape[0]
        best: torch.Tensor | None = None
        # Python 层的循环在导出时会被展开成固定条数的子图，
        # 所以中间张量恒为 [cells, chunk_rows]，不随库行数增长。
        for start in range(0, rows, self.chunk_rows):
            block = self.bank[start : start + self.chunk_rows]
            similarity = tokens @ block.t()
            local_max = similarity.max(dim=1).values
            best = local_max if best is None else torch.maximum(best, local_max)
        assert best is not None  # 库非空，循环至少执行一次

        # 归一化 L2 距离 / 2 与 1 - cos 等价（上游 knn_metric="L2_normalized"）。
        distances = 1.0 - best.clamp(-1.0, 1.0)
        anomaly_map = distances.reshape(1, 1, self.grid_h, self.grid_w)
        score = torch.topk(distances, self.top_k).values.mean().reshape(1, 1)
        return anomaly_map, score


@torch.inference_mode()
def predict_reference(
    model: nn.Module,
    batch: torch.Tensor,
    bank: torch.Tensor,
    grid_h: int,
    grid_w: int,
    top_ratio: float = TOP_RATIO,
    chunk_rows: int = DEFAULT_BANK_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """纯 PyTorch 参考实现：分块最大相似度 + top-1% 均值。

    与 `BakedAnomalyDINO.forward` 应当逐元素一致；两者分开写是为了让导出图
    出问题时能立刻分辨「是算法错了」还是「是导出错了」。
    """
    tokens = extract_patch_tokens(model, batch)[0]
    tokens = F.normalize(tokens.float(), dim=-1)
    best: torch.Tensor | None = None
    for start in range(0, bank.shape[0], chunk_rows):
        block = bank[start : start + chunk_rows]
        local_max = (tokens @ block.t()).max(dim=1).values
        best = local_max if best is None else torch.maximum(best, local_max)
    assert best is not None
    distances = 1.0 - best.clamp(-1.0, 1.0)
    anomaly_map = distances.reshape(1, 1, grid_h, grid_w)
    score = torch.topk(distances, top1p_count(grid_h * grid_w, top_ratio)).values.mean().reshape(1, 1)
    return anomaly_map, score


def preprocess_like_dll(images: Iterable, resolution: int) -> torch.Tensor:
    """按 `anomalib_infer.dll` 的口径做预处理：拉伸 + /255 + ImageNet 归一。

    DLL 用的是 `cv::resize` 拉伸（`anomalib_infer_c.cpp:367`），不是上游那种
    「短边缩放到 448」的等比例缩放。标定必须复刻这个口径，否则阈值与线上判定
    会对不上——这是本项目最容易踩的坑，所以单独一个函数、单独一条注释。

    `images` 的元素是 uint8 的 RGB array（H, W, 3）。
    """
    import numpy as np

    tensors = []
    for image in images:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"期望 RGB 三通道图像，实际形状 {array.shape}")
        # 用 torch 做拉伸，保证与 DLL 的双线性口径一致（不引入 PIL 的 bicubic）。
        tensor = torch.from_numpy(array.astype("float32") / 255.0).permute(2, 0, 1)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0]
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors, dim=0)
