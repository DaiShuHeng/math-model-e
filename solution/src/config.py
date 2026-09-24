"""Central configuration: paths, constants, empirical augmentation parameters.

All empirical corruption parameters come from the forensic audit of 附件3
(analysis/vocab_cache/adv/zero_row.damage_sets.csv, adjudicated 2026-09-24).
"""
from pathlib import Path

ROOT = Path("/home/daishuheng/math_competition")
DATA_DIR = ROOT / "E题" / "E题数据"
ALIGNED_PKL = DATA_DIR / "附件2-数据集特征文件" / "aligned_50.pkl"
ATT3_DIR = DATA_DIR / "附件3-模态缺失特征样本" / "对齐版本"
ATT4_DIR = (
    DATA_DIR / "附件4-可解释专项视频样本与特征文件"
    / "附件4-可解释专项视频样本与特征文件" / "对齐版本"
)
BERT_DIR = ROOT / "models" / "bert-tiny"

SOLUTION = ROOT / "solution"
CACHE = SOLUTION / "cache"
WEIGHTS = SOLUTION / "weights"
RESULTS = SOLUTION / "results"
LOGS = SOLUTION / "logs"

# ---- tokenizer interface (verified 2026-09-24, see analysis/词表与接口核验结论.md) ----
MAX_LEN = 50
PAD_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102

# ---- label mapping (audited): 0=Negative, 1=Neutral, 2=Positive ----
N_CLASSES = 3
CLASS_NAMES = ["Negative", "Neutral", "Positive"]

# ---- augmentation: empirical rate pool from 附件3 (30 files, 3 zeros included) ----
P_FILE = 0.9
EMPIRICAL_RATES = [
    0.0, 0.1786, 0.1562, 0.0645, 0.0, 0.12, 0.1429, 0.1333, 0.1429, 0.0,
    0.2857, 0.1538, 0.1053, 0.087, 0.1111, 0.1176, 0.3636, 0.1765, 0.2917,
    0.2917, 0.3333, 0.2353, 0.3077, 0.25, 0.2222, 0.2727, 0.381, 0.3333,
    0.4444, 0.5,
]

# ---- training defaults ----
SEED = 2026
DEVICE = "cuda:0"
