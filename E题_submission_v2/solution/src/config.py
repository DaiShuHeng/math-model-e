"""Revised protocol: predefined continuous blocks; no special-test fitted rates.
Changing this protocol requires NEW training and NEW validation results.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "solution"
DATA_DIR = Path(os.environ.get("MATH_E_DATA", str(ROOT / "E题" / "E题数据"))).resolve()
ALIGNED_PKL = DATA_DIR / "附件2-数据集特征文件" / "aligned_50.pkl"
ATT3_DIR = DATA_DIR / "附件3-模态缺失特征样本" / "对齐版本"
ATT4_DIR = DATA_DIR / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件" / "对齐版本"
BERT_DIR = Path(os.environ.get("MATH_E_BERT", str(ROOT / "models" / "bert-tiny")))
CACHE = SOLUTION / "cache"
WEIGHTS = SOLUTION / "weights"
RESULTS = SOLUTION / "results"
LOGS = SOLUTION / "logs"
MAX_LEN = 50
PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 100, 101, 102
N_CLASSES = 3
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
# Design values; chosen before any new runs, NOT estimated from attachment 3.
P_FILE = 0.7
PREDEFINED_RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
PROTOCOL_VERSION = "continuous_blocks_v3_no_special_fit"
SEED = 2026
DEVICE = os.environ.get("MATH_E_DEVICE", "cuda:0")
