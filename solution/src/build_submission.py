"""Build the anonymized, size-budgeted submission bundle.

Layout inside the bundle mirrors the working tree (solution/ and
math_model_e_optimized/ as siblings under the bundle root), so each absolute
/home/... literal is rewritten to the equivalent _E_ROOT expression:

    _E_ROOT = Path(__file__).resolve().parents[2]   # src -> solution -> root

Excludes: weights/, cache/, __pycache__, .vscode (teammate OneDrive path), .env.
Per-file size cap 2 MB keeps the small p2/p3 head checkpoints (~1.9 MB) and all
code/results/data features, drops nothing else of value.

Run:  cd solution && python -m src.build_submission
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SOL = Path(__file__).resolve().parents[1]
MC = SOL.parent
OUT = MC / "E题_submission"

INCLUDE_SOL = ["src", "tests", "results", "logs", "paper", "README.md", "run_v2_queue.sh"]
SOL_DATA_SUB = ["p1_features_v3", "p1_summary_v3.csv"]
INCLUDE_TM = ["code", "results", "paper", "models"]
TM_DATA_SUB = ["alignment", "p1_features", "p1_summary_v2.csv"]
SIZE_CAP = 2 * 1024 * 1024

HELPER = '_E_ROOT = Path(__file__).resolve().parents[2]\n'

# exact code-literal rewrites (order matters: specific before generic)
PY_REWRITES = [
    ('Path("/home/daishuheng/math_competition/math_model_e_optimized")',
     '_E_ROOT / "math_model_e_optimized"'),
    ('"/home/daishuheng/math_competition/math_model_e_optimized/data/p1_summary_v2.csv"',
     'str(_E_ROOT / "math_model_e_optimized" / "data" / "p1_summary_v2.csv")'),
    ('"/home/daishuheng/math_competition/models/bert-base-uncased"',
     'str(_E_ROOT / "models" / "bert-base-uncased")'),
    ('"/home/daishuheng/math_competition/models/face_detection_yunet_2023mar.onnx"',
     'str(_E_ROOT / "models" / "face_detection_yunet_2023mar.onnx")'),
    ('"/home/daishuheng/math_competition/E题/E题数据"',
     'str(_E_ROOT / "E题" / "E题数据")'),
    ('ROOT = Path("/home/daishuheng/math_competition")', 'ROOT = _E_ROOT'),
    # docstring / comment residue (after code literals are gone)
    ('/home/daishuheng/math_competition/E题/E题数据', '$E_ROOT/E题/E题数据'),
    ('/home/daishuheng/math_competition', '$E_ROOT'),
]
SH_REWRITES = [
    ('/home/daishuheng/miniconda3/envs/mosei/bin/python', 'python'),
    ('/home/daishuheng/math_competition/solution', '$E_ROOT/solution'),
    ('/home/daishuheng/math_competition/models', '$E_ROOT/models'),
]


def copy_tree(src: Path, dst: Path, data_sub: list[str] | None = None):
    if src.is_file():
        shutil.copy2(src, dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    skip = {"__pycache__", ".vscode", ".env", ".pytest_cache", ".ipynb_checkpoints",
            "build_submission.py"}
    for p in sorted(src.iterdir()):
        if p.name in skip:
            continue
        if data_sub is not None and src.name == "data":
            if p.name in data_sub:
                copy_tree(p, dst / p.name)
            continue
        if p.is_dir():
            copy_tree(p, dst / p.name)
        elif p.stat().st_size < SIZE_CAP:
            shutil.copy2(p, dst / p.name)


def rewrite_py(txt: str) -> str:
    for old, new in PY_REWRITES:
        txt = txt.replace(old, new)
    if "_E_ROOT" in txt and HELPER not in txt:
        lines = txt.splitlines(keepends=True)
        last_import = max(i for i, l in enumerate(lines[:80])
                          if l.startswith(("import ", "from ")))
        lines.insert(last_import + 1, HELPER)
        txt = "".join(lines)
    return txt


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "solution").mkdir(parents=True)
    (OUT / "math_model_e_optimized").mkdir(parents=True)
    shutil.copy2(SOL / "提交说明.md", OUT / "提交说明.md")

    for name in INCLUDE_SOL:
        copy_tree(SOL / name, OUT / "solution" / name,
                  data_sub=SOL_DATA_SUB if name == "data" else None)
    for name in INCLUDE_TM:
        copy_tree(MC / "math_model_e_optimized" / name, OUT / "math_model_e_optimized" / name,
                  data_sub=TM_DATA_SUB if name == "data" else None)

    n_py = 0
    for f in sorted((OUT / "solution").rglob("*.py")):
        t = f.read_text(encoding="utf-8")
        if "daishuheng" in t:
            f.write_text(rewrite_py(t), encoding="utf-8")
            n_py += 1
    for f in sorted((OUT / "math_model_e_optimized").rglob("*.py")):
        t = f.read_text(encoding="utf-8")
        if "daishuheng" in t or "OneDrive" in t:
            f.write_text(rewrite_py(t), encoding="utf-8")
            n_py += 1
    sh = OUT / "solution" / "run_v2_queue.sh"
    if sh.exists():
        t = sh.read_text(encoding="utf-8")
        for old, new in SH_REWRITES:
            t = t.replace(old, new)
        sh.write_text(t, encoding="utf-8")

    # console logs embed home paths in command lines — sanitize textually
    for f in OUT.rglob("*.log"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        f.write_text(t.replace("/home/daishuheng", "$HOME"), encoding="utf-8")

    # ---- verification: syntax, anonymity, size ----
    r = subprocess.run([sys.executable, "-m", "compileall", "-q", str(OUT)],
                       capture_output=True, text=True)
    print("compileall:", "OK" if r.returncode == 0 else "FAIL\n" + r.stderr[:800])

    bad = []
    for f in OUT.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".md", ".json", ".csv", ".sh", ".txt", ".log"}:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if "daishuheng" in t or "OneDrive" in t:
                bad.append(str(f.relative_to(OUT)))
    print("anonymity:", "CLEAN" if not bad else f"LEAKS: {bad}")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"rewrote {n_py} py files; bundle: {total/1e6:.1f} MB "
          f"({'OK <50MB' if total < 50e6 else 'OVER 50MB!'})")
    print(f"bundle at {OUT}")


if __name__ == "__main__":
    main()
