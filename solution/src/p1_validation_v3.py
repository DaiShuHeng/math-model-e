"""Grouped-CV linear probe on the v3 (YuNet-enhanced) p1 features.

Same protocol as the teammate's p1_validation.py (StratifiedGroupKFold(5) by
source video, 3 seeds, StandardScaler+LogReg/Ridge inside folds) but reading
solution/data/p1_features_v3 + p1_summary_v3. Purpose: show the YuNet vision
fix improves not just coverage (39->4 zero-vision videos) but also usable
signal for downstream prediction.

Run:
  cd solution && MATH_E_DATA=$E_ROOT/E题/E题数据 \
    python -m src.p1_validation_v3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_E_ROOT = Path(__file__).resolve().parents[2]
SOL = Path(__file__).resolve().parents[1]
TM = _E_ROOT / "math_model_e_optimized"
os.environ.setdefault("MATH_E_DATA", str(_E_ROOT / "E题" / "E题数据"))
sys.path.insert(0, str(TM / "code"))

import config as C                     # noqa: E402
import trainer as T                    # noqa: E402

MODS = ("text", "audio", "vision")


def probe(summary_csv: str, feat_dir: Path):
    df = pd.read_csv(summary_csv, dtype={"clip_id": str})
    y = df.label.to_numpy()
    yc = np.sign(y).astype(int) + 1
    groups = df.video_id.to_numpy()
    feats = {m: [] for m in MODS}
    for sid in df.sample_id:
        with np.load(feat_dir / f'{sid.replace(C.SEP, "__")}.npz') as z:
            for m in MODS:
                mask = z[f"mask_{m}"]
                feats[m].append((z[m] * mask[:, None]).sum(0) / max(mask.sum(), 1))
    rows = []
    for name, mods in [("text", ["text"]), ("audio", ["audio"]),
                       ("vision", ["vision"]), ("fusion", list(MODS))]:
        X = np.concatenate([np.stack(feats[m]) for m in mods], 1)
        for seed in (2026, 2027, 2028):
            pc = np.zeros(len(y), int)
            pr = np.zeros(len(y))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, yc, groups):
                assert not set(groups[tr]) & set(groups[te])
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=2000, class_weight="balanced"))
                reg = make_pipeline(StandardScaler(), Ridge(alpha=10.))
                pc[te] = clf.fit(X[tr], yc[tr]).predict(X[te])
                pr[te] = reg.fit(X[tr], y[tr]).predict(X[te]).clip(-3, 3)
            rows.append({"feature": name, "seed": seed, **T.compute_metrics(yc, pc, y, pr)})
    return pd.DataFrame(rows)


def main():
    v3 = probe(SOL / "data" / "p1_summary_v3.csv", SOL / "data" / "p1_features_v3")
    v3.to_csv(SOL / "results" / "p1_grouped_cv_v3.csv", index=False, encoding="utf-8-sig")
    print("=== v3 (YuNet vision) ===")
    print(v3.groupby("feature")[["acc", "f1_macro", "mae", "pearson"]].agg(["mean", "std"]).round(4).to_string())

    # regenerate v2 numbers under the SAME code path / current artifacts: the
    # teammate's results/p1_grouped_cv.csv predates the physical-bin refine
    # (its text/audio rows differ from what their current NPZs produce), so
    # it must not be used for a v2-vs-v3 comparison.
    v2 = probe(TM / "data" / "p1_summary_v2.csv", TM / "data" / "p1_features")
    v2.to_csv(SOL / "results" / "p1_grouped_cv_v2_regen.csv", index=False, encoding="utf-8-sig")
    print("\n=== v2 (Haar vision, current artifacts, same code path) ===")
    print(v2.groupby("feature")[["acc", "f1_macro", "mae", "pearson"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
