"""Audit the actual missingness structure of the provided feature files.

Read-only: prints/serialises the statistics quoted in docs/缺失结构审计.md.
No labels are used, no test-set inputs are fed to any model, and nothing is
written outside the requested --out path.

Run (from `solution/`):
  python -m src.audit_missing_structure --out ../logs/missing_structure_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .q2_features import load_aligned_splits, load_attachment3, load_attachment4, read_pickle


def _stats(split, tok: np.ndarray | None = None) -> dict:
    a_obs = split.a_obs.astype(bool)
    v_obs = split.v_obs.astype(bool)
    t_obs = split.t_obs.astype(bool)
    n, T = a_obs.shape
    ar = np.arange(T)[None, :]
    span = t_obs if tok is None else (ar < tok[:, None])
    out = {
        "n": int(n),
        "audio_zero_outside_span": float((~a_obs)[~span].mean()) if (~span).any() else None,
        "vision_zero_outside_span": float((~v_obs)[~span].mean()) if (~span).any() else None,
        "audio_obs_eq_span": float((a_obs == span).mean()),
        "vision_obs_eq_span": float((v_obs == span).mean()),
        "audio_eq_vision": float((a_obs == v_obs).mean()),
        "audio_obs_rate": float(a_obs.mean()),
        "vision_obs_rate": float(v_obs.mean()),
        "text_obs_rate": float(t_obs.mean()),
    }
    inside = span
    if inside.any():
        out["audio_obs_rate_inside_span"] = float(a_obs[inside].mean())
        out["vision_obs_rate_inside_span"] = float(v_obs[inside].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(config.LOGS / "missing_structure_audit.json"))
    args = ap.parse_args()
    config.LOGS.mkdir(parents=True, exist_ok=True)

    res = {}
    raw = read_pickle(config.ALIGNED_PKL)
    for s in ("train", "valid", "test"):
        sp = load_aligned_splits((s,))[s]
        tok = np.asarray(raw[s]["text_bert"])[:, 1, :].sum(1)
        res[s] = _stats(sp, tok)
    for name, loader in (("att3", load_attachment3), ("att4", load_attachment4)):
        sp = loader()
        # token span for the attachments comes from their own text_bert
        toks = []
        for p in sorted((config.ATT3_DIR if name == "att3" else config.ATT4_DIR).glob("*.pkl")):
            d = read_pickle(p)
            d = d["test"] if "test" in d and isinstance(d["test"], dict) else d
            toks.append(int(np.asarray(d["text_bert"])[1].sum()))
        res[name] = _stats(sp, np.asarray(toks))
        # per-sample detail for the specialised sets
        if name == "att3":
            res["att3_per_sample"] = [
                {"id": sp.sample_id[i],
                 "audio_obs": int(sp.a_obs[i].sum()), "vision_obs": int(sp.v_obs[i].sum()),
                 "token_span": int(toks[i]),
                 "audio_equals_vision": bool(np.array_equal(sp.a_obs[i], sp.v_obs[i]))}
                for i in range(len(sp))]

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "att3_per_sample"},
                     ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
