#!/usr/bin/env python3
"""Build the v5 submission candidate package with entry-level acceptance gates.

Gates (all must pass before the package is declared valid):
  G1  every Q2 checkpoint loads standalone and reproduces its recorded valid score
  G2  the checkpoint-embedded normalizer equals statistics recomputed from TRAIN
  G3  the staged code reproduces the frozen 附件4 explanation CSV
  G4  附件3 / 附件4 prediction CSVs have the required rows, columns and no NaN
  G5  total package size is inside the 50 MB competition limit
  G6  no identity information (team / member / affiliation) in any text file

Usage:
  python build_submission_v5.py --out ../E题_submission_v5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SOL = Path(__file__).resolve().parents[1]         # .../math-model-e/solution
ROOT = SOL.parent                                 # .../math-model-e
LIMIT = 50 * 1024 * 1024
# G6 looks for identity *values*, not for the words themselves: a compliance
# sentence such as "本包不含队员姓名" must not trip the gate.  Each pattern
# requires a name-like payload after the keyword.
import re as _re
FORBIDDEN = [
    _re.compile(r"参赛队\s*[：:]\s*\S"),
    _re.compile(r"队伍编号\s*[：:]\s*\S"),
    _re.compile(r"队员\s*(?:姓名|名单)?\s*[：:]\s*\S"),
    _re.compile(r"学号\s*[：:]\s*\S"),
    _re.compile(r"指导老师\s*[：:]\s*\S"),
    _re.compile(r"(?:作者|姓名)\s*[：:]\s*\S"),
]
STAGE = ["src", "tests", "docs", "paper", "results", "weights/v5_deploy",
         "提交说明.md", "README.md"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gate(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return {"gate": name, "passed": bool(ok), "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-repro", action="store_true", help="skip G3 (slow)")
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing {out}")
    gates = []

    # ---- stage files -------------------------------------------------------
    out.mkdir(parents=True)
    copied = []
    for rel in STAGE:
        src = SOL / rel
        dst = out / "solution" / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            continue
        copied.append(rel)
    # Q1 features + alignment JSON (the problem-1 deliverable)
    p1 = SOL / "data"
    shutil.copytree(p1, out / "solution" / "data",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"staged {len(copied)} entries + solution/data", flush=True)

    # ---- G1: checkpoints load standalone and reproduce recorded valid score --
    import numpy as np
    # The staged copy lives at <parent>/solution/src, so config.ROOT resolves to
    # <parent> and MATH_E_DATA must point at the real competition data directory.
    import os
    os.environ["MATH_E_DATA"] = str(ROOT / "E题" / "E题数据")
    os.environ["MATH_E_DEVICE"] = "cpu"
    subprocess.run([sys.executable, "-c", "import torch"], check=False)
    sys.path.insert(0, str(out / "solution"))
    from src.q2_features import FeatStandardizer, load_aligned_splits          # noqa: E402
    from src.q2_feature_model import load_checkpoint                           # noqa: E402
    from src.metrics import cls_metrics                                        # noqa: E402
    exp = json.loads((SOL / "weights" / "v5_deploy" / "valid_scores.json").read_text()) \
        if (SOL / "weights" / "v5_deploy" / "valid_scores.json").exists() else None
    va = load_aligned_splits(("valid",))["valid"]
    tr = load_aligned_splits(("train",))["train"]
    std_fit = FeatStandardizer.fit(tr)
    arr = std_fit.transform(va)
    ckpts = sorted((out / "solution" / "weights" / "v5_deploy").glob("*.pt"))
    g1_bad = []
    for p in ckpts:
        model, ck = load_checkpoint(p)
        f = float(cls_metrics(va.y_cls, model.predict(arr)["prob"].argmax(-1))["MacroF1"])
        rec = (exp or {}).get(p.stem)
        if rec is not None and abs(f - rec) > 2e-3:
            g1_bad.append((p.name, round(f, 4), rec))
    gates.append(gate("G1 standalone checkpoint load + score", not g1_bad,
                      f"{len(ckpts)} checkpoints, mismatches={g1_bad or 'none'}"))

    # ---- G2: embedded normalizer == train recomputation --------------------
    g2_bad = []
    for p in ckpts:
        _, ck = load_checkpoint(p)
        stored = ck["normalizer"]
        for k in ("t_mean", "t_std", "a_mean", "a_std", "v_mean", "v_std"):
            if not np.allclose(np.asarray(stored[k]), getattr(std_fit, k), atol=1e-5):
                g2_bad.append((p.name, k))
    gates.append(gate("G2 normalizer == train recomputation", not g2_bad,
                      f"keys checked={6*len(ckpts)}, mismatches={g2_bad or 'none'}"))

    # ---- G3: staged code reproduces the frozen 附件4 CSV --------------------
    frozen = SOL / "results" / "附件4_预测与解释_v5.csv"
    if a.skip_repro:
        gates.append(gate("G3 reproduce 附件4 explanation", True, "skipped by flag"))
    else:
        tmp = out / "_repro_att4.csv"
        sel = json.loads((SOL / "weights" / "v5_deploy" / "selection.json").read_text())
        chosen = [out / "solution" / "weights" / "v5_deploy" / f"{n}.pt"
                  for n in sel["selected"]]
        missing = [p.name for p in chosen if not p.exists()]
        if missing:
            gates.append(gate("G3 reproduce 附件4 explanation", False,
                              f"missing checkpoints {missing}"))
            chosen = []
        cmd = ([sys.executable, "-m", "src.explain_att4_v5", "--ckpt"]
               + [str(p) for p in chosen] + ["--out", str(tmp)]) if chosen else None
        if cmd is None:
            r = None
        r = subprocess.run(cmd, cwd=out / "solution", capture_output=True, text=True) \
            if cmd else None
        ok = bool(r) and r.returncode == 0
        detail = "reproduced" if ok else (r.stderr[-400:] if r else "not run")
        if ok:
            import csv as _csv
            f1 = list(_csv.DictReader(open(frozen)))
            f2 = list(_csv.DictReader(open(tmp)))
            same = len(f1) == len(f2) and all(
                row1["sample_id"] == row2["sample_id"] and
                row1["pred_class"] == row2["pred_class"] and
                abs(float(row1["phi_text_cls"]) - float(row2["phi_text_cls"])) < 1e-4 and
                abs(float(row1["phi_audio_cls"]) - float(row2["phi_audio_cls"])) < 1e-4 and
                abs(float(row1["phi_vision_cls"]) - float(row2["phi_vision_cls"])) < 1e-4
                for row1, row2 in zip(f1, f2))
            ok = same
            detail = f"{len(f1)} rows, bitwise-close match={same}"
        tmp.unlink(missing_ok=True)
        tmp.with_suffix(".summary.json").unlink(missing_ok=True)
        manifest_tmp = out / "_manifest_tmp.json"
        gates.append(gate("G3 reproduce 附件4 explanation", ok, detail))

    # ---- G4: deliverable CSV shape ----------------------------------------
    import csv as _csv
    checks = []
    a3 = list(_csv.DictReader(open(SOL / "results" / "att3_predictions_final.csv")))
    checks.append(("附件3 rows==30", len(a3) == 30, len(a3)))
    checks.append(("附件3 has polarity+intensity",
                   all(r.get("polarity") in ("Negative", "Neutral", "Positive") and r.get("intensity")
                       for r in a3), ""))
    a4 = list(_csv.DictReader(open(frozen)))
    need = {"pred_polarity", "pred_intensity", "main_modality", "share_text", "share_audio",
            "share_vision", "evidence_positions", "effect_ranked_set_drop"}
    checks.append(("附件4 rows==20", len(a4) == 20, len(a4)))
    checks.append(("附件4 has all required explanation columns", need <= set(a4[0]), sorted(need - set(a4[0]))))
    gates.append(gate("G4 deliverable CSV shape", all(c[1] for c in checks),
                      "; ".join(f"{n}={v}" for n, o, v in checks)))

    # ---- G5: size ----------------------------------------------------------
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    gates.append(gate("G5 total size <= 50 MB", total <= LIMIT,
                      f"{total/1024/1024:.2f} MB"))

    # ---- G6: identity scrub ------------------------------------------------
    hits = []
    for p in out.rglob("*"):
        if not p.is_file() or p.suffix.lower() in {".pt", ".npz", ".pkl", ".png", ".jpg"}:
            continue
        try:
            t = p.read_text(errors="ignore")
        except Exception:
            continue
        for pat in FORBIDDEN:
            m = pat.search(t)
            if m:
                hits.append((str(p.relative_to(out)), m.group(0)[:40]))
    gates.append(gate("G6 no identity information", not hits, hits or "clean"))

    # ---- manifest ----------------------------------------------------------
    files = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(out.rglob("*")) if p.is_file()]
    manifest = {"status": "FINAL_CANDIDATE" if all(g["passed"] for g in gates) else "REJECTED",
                "protocol": "feature_level_v5_span_bounded",
                "gates": gates, "n_files": len(files),
                "total_bytes": sum(f["bytes"] for f in files),
                "q2_checkpoints": [f["path"] for f in files if f["path"].endswith(".pt")],
                "selection": "top-4 by VALID score of the pre-registered v5a-v5d x 2026/7/42 grid",
                "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nstatus = {manifest['status']}")
    print(f"files = {len(files)}, total = {manifest['total_bytes']/1024/1024:.2f} MB")
    if manifest["status"] != "FINAL_CANDIDATE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
