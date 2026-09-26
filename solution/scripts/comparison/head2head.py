"""Head-to-head on the SAME labeled test split (aligned_50.pkl 'test', n=727).

Everything is evaluated with one shared metric implementation and one shared
data path, so the comparison is apples-to-apples:
  * teammate p2/p3  : their 3-seed ensembles + their validation-fitted log-prob bias
  * mine v5         : my 4-checkpoint deployment subset, and the 12-model mean
Also reports what happens to MY model under the same bias-calibration step.
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

torch.set_num_threads(6)
PKG = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/E题_最终优化版")
MY = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/math-model-e/solution")
DATA = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/math-model-e/E题/E题数据")
os.environ["MATH_E_DATA"] = str(DATA)
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(PKG / "code"))
sys.path.insert(0, str(MY))

import config as C                      # noqa: E402   (their config)
import io_utils as U                    # noqa: E402   (their loader)
from inference import load_model        # noqa: E402   (their model)
from src.q2_features import FeatStandardizer, load_aligned_splits   # noqa: E402
from src.q2_feature_model import load_checkpoint                    # noqa: E402


def metrics(y, p, yr=None, pr=None):
    m = {"acc": float(accuracy_score(y, p)),
         "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
         "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0))}
    if yr is not None:
        m["mae"] = float(np.mean(np.abs(yr - pr)))
        m["pearson"] = float(np.corrcoef(yr, pr)[0, 1])
    return m


def best_bias(prob, y):
    best = None
    for bn, bu in itertools.product([0., -.2, .2], repeat=2):
        b = np.array([bn, bu, 0.])
        p = (np.log(prob.clip(1e-9)) + b).argmax(1)
        s = f1_score(y, p, average="macro")
        if best is None or s > best[0] + 1e-12:
            best = (s, b)
    return best[1]


def apply_bias(prob, b):
    lp = np.log(prob.clip(1e-9)) + np.asarray(b)
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


# ---------------- shared labels ----------------
a2 = U.load_a2_all(splits=("valid", "test"))
y_va = np.array([s.label_cls for s in a2["valid"]])
y_te = np.array([s.label_cls for s in a2["test"]])
r_va = np.array([s.label_reg for s in a2["valid"]])
r_te = np.array([s.label_reg for s in a2["test"]])

# ---------------- MY model ----------------
std = FeatStandardizer.fit(load_aligned_splits(("train",))["train"])
sp_va = load_aligned_splits(("valid",))["valid"]
sp_te = load_aligned_splits(("test",))["test"]
arr_va, arr_te = std.transform(sp_va), std.transform(sp_te)
CK = sorted((MY / "weights" / "v5_deploy").glob("*.pt"))
sel = json.loads((MY / "weights" / "v5_deploy" / "selection.json").read_text())["selected"]
print("my deployment subset:", sel)

def mine(keys):
    P = np.mean([load_checkpoint(str(MY / "weights" / "v5_deploy" / f"{k}.pt"))[0].predict(arr_te)["prob"] for k in keys], 0)
    R = np.mean([load_checkpoint(str(MY / "weights" / "v5_deploy" / f"{k}.pt"))[0].predict(arr_te)["reg"] for k in keys], 0)
    Pv = np.mean([load_checkpoint(str(MY / "weights" / "v5_deploy" / f"{k}.pt"))[0].predict(arr_va)["prob"] for k in keys], 0)
    return P, R, Pv

results = {}

# teammate
for tag in ("p2", "p3"):
    net = load_model(tag)
    from inference import predict as tpredict
    va = tpredict(net, a2["valid"]); te = tpredict(net, a2["test"])
    b = best_bias(va["prob"], y_va)
    results[f"teammate_{tag}"] = {
        "valid_raw": metrics(y_va, va["p_cls"], r_va, va["p_reg"]),
        "valid_cal": metrics(y_va, apply_bias(va["prob"], b).argmax(1)),
        "test_raw": metrics(y_te, te["p_cls"], r_te, te["p_reg"]),
        "test_cal": metrics(y_te, apply_bias(te["prob"], b).argmax(1)),
        "bias": b.tolist()}

# mine
for label, keys in (("v5_deploy4", sel), ("v5_best_single", ["c3_s2026"]),
                    ("v5_all12", [p.stem for p in CK])):
    P, R, Pv = mine(keys)
    b = best_bias(Pv, y_va)                    # bias fitted on VALID only
    results[f"mine_{label}"] = {
        "valid_raw": metrics(y_va, Pv.argmax(1), r_va, np.mean([load_checkpoint(str(MY/'weights'/'v5_deploy'/f'{k}.pt'))[0].predict(arr_va)["reg"] for k in keys], 0)),
        "valid_cal": metrics(y_va, apply_bias(Pv, b).argmax(1)),
        "test_raw": metrics(y_te, P.argmax(1), r_te, R),
        "test_cal": metrics(y_te, apply_bias(P, b).argmax(1)),
        "bias": b.tolist()}

print("\n" + "=" * 96)
print(f"{'model':22s} {'valid RAW':>18s} {'valid CAL':>18s} {'TEST RAW':>18s} {'TEST CAL':>18s}")
print(f"{'':22s} {'acc / macroF1':>18s} {'acc / macroF1':>18s} {'acc / macroF1':>18s} {'acc / macroF1':>18s}")
print("=" * 96)
for k, v in results.items():
    row = f"{k:22s}"
    for key in ("valid_raw", "valid_cal", "test_raw", "test_cal"):
        m = v[key]
        row += f"{m['acc']:.4f} / {m['f1_macro']:.4f}".rjust(19)
    print(row)
print("=" * 96)
print("\nMAE / Pearson on TEST (raw):")
for k, v in results.items():
    m = v["test_raw"]
    print(f"  {k:22s} MAE {m['mae']:.4f}  pearson {m['pearson']:.4f}")
print("\nvalidation-fitted bias:")
for k, v in results.items():
    print(f"  {k:22s} {np.round(v['bias'],2).tolist()}")
Path("/tmp/eexp/head2head.json").write_text(json.dumps(results, indent=2))
print("\nwritten /tmp/eexp/head2head.json")
