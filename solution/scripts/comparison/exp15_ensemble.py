"""Exp15: cross-team probability ensembling.

Members available without retraining or test access:
  * my v5 feature-level models (12 checkpoints, 4 cfg x 3 seeds)
  * the teammate's p2 and p3 3-seed ensembles on the SAME aligned features
Every combination below is SELECTED on VALID only; TEST is reporting only.
"""
import itertools, json, os, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

torch.set_num_threads(6)
PKG = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/E题_最终优化版")
S = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/math-model-e/solution")
os.environ["MATH_E_DATA"] = str(S.parent / "E题" / "E题数据")
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(PKG / "code"))
sys.path.insert(0, str(S))

import io_utils as U                              # noqa: E402
from inference import load_model, predict         # noqa: E402
from src.q2_features import FeatStandardizer, load_aligned_splits   # noqa: E402
from src.q2_feature_model import load_checkpoint                    # noqa: E402

a2 = U.load_a2_all(splits=("valid", "test"))
yva = np.array([s.label_cls for s in a2["valid"]])
yte = np.array([s.label_cls for s in a2["test"]])

std = FeatStandardizer.fit(load_aligned_splits(("train",))["train"])
va = std.transform(load_aligned_splits(("valid",))["valid"])
te = std.transform(load_aligned_splits(("test",))["test"])

P = {}
for p in sorted((S / "weights" / "v5_deploy").glob("*.pt")):
    m, _ = load_checkpoint(str(p))
    P[f"v5_{p.stem}"] = (m.predict(va)["prob"], m.predict(te)["prob"])
for tag in ("p2", "p3"):
    net = load_model(tag)
    P[f"tm_{tag}"] = (predict(net, a2["valid"])["prob"], predict(net, a2["test"])["prob"])
print("members:", len(P))

GRID = list(itertools.product([0., -.2, .2, -.4, .4], repeat=2))
def f1(y, p, b=(0., 0.)):
    lp = np.log(p.clip(1e-9)) + np.array([b[0], b[1], 0.])
    return f1_score(y, lp.argmax(1), average="macro")
def acc(y, p, b=(0., 0.)):
    lp = np.log(p.clip(1e-9)) + np.array([b[0], b[1], 0.])
    return accuracy_score(y, lp.argmax(1))
def pick_bias(p):
    return max(GRID, key=lambda b: f1(yva, p, b))

names = list(P)
def ens(keys):
    return (np.mean([P[k][0] for k in keys], 0), np.mean([P[k][1] for k in keys], 0))

# greedy forward selection on VALID, allowing bias calibration
def greedy(pool, max_k):
    chosen, cur, best_v = [], None, -1
    while len(chosen) < max_k:
        best = None
        for k in pool:
            if k in chosen:
                continue
            pv, _ = ens(chosen + [k])
            b = pick_bias(pv)
            s = f1(yva, pv, b)
            if best is None or s > best[0]:
                best = (s, k)
        if best[0] <= best_v + 1e-6:
            break
        best_v = best[0]; chosen.append(best[1])
    return chosen, best_v

rows = []
def report(label, keys):
    pv, pt = ens(keys)
    b = pick_bias(pv)
    rows.append(dict(label=label, n=len(keys), bias=[float(x) for x in b],
                     valid_raw=f1(yva, pv), valid_cal=f1(yva, pv, b),
                     test_raw=f1(yte, pt), test_cal=f1(yte, pt, b),
                     test_acc_raw=acc(yte, pt), test_acc_cal=acc(yte, pt, b),
                     members=keys))

report("mine_v5_top4", ["v5_c0_s2026", "v5_c2_s2026", "v5_c2_s42", "v5_c3_s2026"])
report("mine_v5_all12", [k for k in names if k.startswith("v5_")])
report("teammate_p2", ["tm_p2"])
report("teammate_p2+p3", ["tm_p2", "tm_p3"])
report("v5_top4 + p2", ["v5_c0_s2026", "v5_c2_s2026", "v5_c2_s42", "v5_c3_s2026", "tm_p2"])
report("v5_top4 + p2 + p3", ["v5_c0_s2026", "v5_c2_s2026", "v5_c2_s42", "v5_c3_s2026", "tm_p2", "tm_p3"])
report("ALL v5_12 + tm_p2 + tm_p3", names)

sel, vs = greedy(names, 8)
report(f"greedy({len(sel)})", sel)
print(f"greedy picked (valid macro-F1 {vs:.4f}): {sel}\n")

hdr = f"{'combination':26s} {'n':>3s} {'bias':>12s} {'valid raw':>10s} {'valid cal':>10s} {'TEST raw':>10s} {'TEST cal':>10s} {'test acc cal':>12s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['label']:26s} {r['n']:3d} {str([round(x,1) for x in r['bias']]):>12s} "
          f"{r['valid_raw']:10.4f} {r['valid_cal']:10.4f} {r['test_raw']:10.4f} {r['test_cal']:10.4f} "
          f"{r['test_acc_cal']:12.4f}")
json.dump(rows, open("/tmp/eexp/exp15_ensemble.json", "w"), indent=2)
print("\nwritten /tmp/eexp/exp15_ensemble.json")
