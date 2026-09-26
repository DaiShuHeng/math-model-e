"""Independent reproduction of the teammate package's reported test metrics.

Uses THEIR code and THEIR checkpoints, on the SHARED competition data. Nothing is
imported from my own pipeline, so the numbers are directly comparable.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

PKG = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/E题_最终优化版")
DATA = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/math-model-e/E题/E题数据")
os.environ["MATH_E_DATA"] = str(DATA)
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(PKG / "code"))

import torch  # noqa: E402
torch.set_num_threads(6)

import config as C          # noqa: E402
import io_utils as U        # noqa: E402
from inference import load_model, predict   # noqa: E402
from evaluate_final import calibrate, calibrated  # noqa: E402

print("A2 pkl:", (C.A2_DIR / "aligned_50.pkl").exists())
print("stats cache:", U.STATS_PATH.exists())

data = U.load_a2_all(splits=("valid", "test"))
print("valid", len(data["valid"]), "test", len(data["test"]))

out = {}
for tag in ("p2", "p3"):
    net = load_model(tag)
    va = predict(net, data["valid"])
    bias = calibrate(va["prob"], va["y_cls"])
    va_cal = calibrated(va, bias)
    te = calibrated(predict(net, data["test"]), bias)
    out[tag] = {
        "bias": [float(b) for b in bias],
        "valid_raw": va["metrics"],
        "valid_cal": va_cal["metrics"],
        "test": te["metrics"],
    }
    v, t = va["metrics"], te["metrics"]
    print(f"\n{tag}: bias={np.round(bias,2).tolist()}")
    print(f"  valid RAW  acc {v['acc']:.4f}  macroF1 {v['f1_macro']:.4f}  MAE {v['mae']:.4f}  r {v['pearson']:.4f}")
    print(f"  valid CAL  acc {va_cal['metrics']['acc']:.4f}  macroF1 {va_cal['metrics']['f1_macro']:.4f}")
    print(f"  TEST       acc {t['acc']:.4f}  macroF1 {t['f1_macro']:.4f}  MAE {t['mae']:.4f}  r {t['pearson']:.4f}")

# compare against their recorded file
rec = json.loads((PKG / "results" / "final_metrics.json").read_text())
print("\n=== reproducibility vs their results/final_metrics.json ===")
for tag in ("p2", "p3"):
    for k in ("acc", "f1_macro", "mae", "pearson"):
        mine = out[tag]["test"][k]
        theirs = rec[tag]["test"][k]
        print(f"  {tag}.test.{k:10s} mine {mine:.6f}  theirs {theirs:.6f}  match {abs(mine-theirs) < 1e-6}")
Path("/tmp/eexp/teammate_repro.json").write_text(json.dumps(out, indent=2))
print("\nwritten /tmp/eexp/teammate_repro.json")
