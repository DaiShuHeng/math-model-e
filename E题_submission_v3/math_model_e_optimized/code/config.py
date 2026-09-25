"""全局环境、路径与超参数配置。

注意：本模块必须在导入 torch / transformers / cv2 之前 import，
因为它负责设置 OpenMP 与 HuggingFace 镜像相关的环境变量。
"""
from __future__ import annotations

import os
import pathlib

# --------------------------------------------------------------------------
# 环境变量（必须在重型库导入前设置）
# --------------------------------------------------------------------------
# Compatibility workaround for the supplied Anaconda/PyTorch environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# huggingface.co 在本机不可达，改用国内镜像站
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
# Point MATH_E_DATA to the directory containing the four attachment folders.
SRC = pathlib.Path(os.environ.get("MATH_E_DATA", str(ROOT / "source_data"))).resolve()

CODE_DIR = ROOT / "code"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
P1_DIR = DATA_DIR / "p1_features"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
PAPER_DIR = ROOT / "paper"
LOG_DIR = ROOT / "logs"

for _d in (DATA_DIR, CACHE_DIR, P1_DIR, MODEL_DIR, RESULT_DIR, FIG_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 原始数据
A1_DIR = SRC / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
A1_LABEL = A1_DIR / "label-100.xlsx"
A2_DIR = SRC / "附件2-数据集特征文件"
A3_ALIGNED_DIR = SRC / "附件3-模态缺失特征样本" / "对齐版本"
A4_ROOT = SRC / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件"
A4_ALIGNED_DIR = A4_ROOT / "对齐版本"
A4_VIDEO_DIR = A4_ALIGNED_DIR / "videos"

# --------------------------------------------------------------------------
# 模态与序列常量（与赛题 aligned 版本保持一致）
# --------------------------------------------------------------------------
N_POS = 50                # 统一对齐位置数
DIM_TEXT = 768            # BERT 文本特征维度
DIM_AUDIO = 74            # 官方语音特征维度；问题1使用自定义同维描述符
DIM_VISION = 35           # 官方视觉特征维度；问题1使用自定义同维描述符
SEP = "$_$"               # 附件2 id 分隔符

MODALITIES = ("text", "audio", "vision")
MOD_DIM = {"text": DIM_TEXT, "audio": DIM_AUDIO, "vision": DIM_VISION}

# --------------------------------------------------------------------------
# 问题1：特征提取参数
# --------------------------------------------------------------------------
ACOUSTIC = dict(
    sample_rate=16000,
    n_fft=400,            # 25 ms
    hop_length=160,       # 10 ms
    n_mels=26,
    n_mfcc=13,
    fmin=50.0,
    fmax=8000.0,
)
FACE = dict(
    detect_scale=0.5,     # 人脸检测前缩放比（加速）
    min_neighbors=5,
    face_size=(56, 40),   # 归一化后人脸区域 (高, 宽)
    grid_rows=7,          # 7 x 5 = 35 个面部网格
    grid_cols=5,
    ref_quantile=0.5,     # 中性参考脸取逐像素中位数
)
BERT_MODEL = os.environ.get("MATH_E_BERT", "bert-base-uncased")
BERT_MAX_LEN = 128

# --------------------------------------------------------------------------
# 模型与训练超参
# --------------------------------------------------------------------------
class ModelCfg:
    d_model = 96
    n_heads = 4
    n_layers = 1
    d_ff = 192
    dropout = 0.25
    n_classes = 3          # 极性：negative / neutral / positive
    reg_range = 3.0        # 情感强度取值范围 [-3, 3]


class TrainCfg:
    seed = 2026
    epochs = 35
    batch_size = 64
    lr = 6e-4
    weight_decay = 1e-4
    warmup_ratio = 0.1
    grad_clip = 1.0
    label_smoothing = 0.05
    lambda_cls = 1.0
    lambda_reg = 0.6       # 回归损失权重
    lambda_rec = 0.1       # Mean loss per observed feature element, not sum over D.
    lambda_comp = 0.0      # 本次实验关闭跨模态互补正则
    early_stop_patience = 7


# --------------------------------------------------------------------------
# 缺失模拟（问题2/3 数据增广与消融）
# --------------------------------------------------------------------------
MISS_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MISS_TYPES = ("audio", "vision", "text", "av", "tv", "ta", "tav")


def model_cfg_dict() -> dict:
    """可 JSON/ pickle 序列化的模型配置（类属性的 mappingproxy 不能直接存盘）。"""
    return {k: getattr(ModelCfg, k) for k in dir(ModelCfg) if not k.startswith("_")}


def train_cfg_dict() -> dict:
    return {k: getattr(TrainCfg, k) for k in dir(TrainCfg) if not k.startswith("_")}


def ensure_dirs() -> None:
    for _d in (DATA_DIR, CACHE_DIR, P1_DIR, MODEL_DIR, RESULT_DIR, FIG_DIR, LOG_DIR):
        _d.mkdir(parents=True, exist_ok=True)
