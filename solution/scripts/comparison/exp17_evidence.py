"""Exp17: whose LOCAL EVIDENCE is better?

Teammate: 附件4 evidence words chosen by p3's evidence attention (beta) -- the
          `首位文本证据词` / `text_evidence` columns of their final CSV.
Mine:     full occlusion sweep -- every word position deleted on its own, ranked
          by the drop in the explained class probability.

Fair test: take each side's chosen positions, delete them together on the SAME
explained model, and measure the class-probability drop. Higher is a better
localisation of what the prediction actually depends on.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(6)
PKG = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/E题_最终优化版")
S = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/math-model-e/solution")
os.environ["MATH_E_DATA"] = str(S.parent / "E题" / "E题数据")
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(PKG / "code"))
sys.path.insert(0, str(S))

import io_utils as U                        # noqa: E402  (their loader -> same tensors)
from inference import load_model, predict   # noqa: E402
from src.q2_features import FeatStandardizer, load_aligned_splits   # noqa: E402
from src.q2_feature_model import load_checkpoint                   # noqa: E402

# --- their explanation table -------------------------------------------------
tm = pd.read_csv(PKG / "results" / "附件4_最终预测与解释.csv", encoding="utf-8-sig")
tm = tm.set_index(tm["sample_id"].astype(str).str.zfill(2))
print("teammate 附件4 rows:", len(tm), "| evidence columns:",
      [c for c in tm.columns if "证据" in c or "evidence" in c][:6])

# --- my deployment ensemble --------------------------------------------------
std = FeatStandardizer.fit(load_aligned_splits(("train",))["train"])
sp = load_attachment4 = None
from src.q2_features import load_attachment4 as _l4      # noqa: E402
sp = _l4()
arr = std.transform(sp)
sel = json.loads((S / "weights" / "v5_deploy" / "selection.json").read_text())["selected"]
MINE = [load_checkpoint(str(S / "weights" / "v5_deploy" / f"{k}.pt"))[0] for k in sel]

# --- their p3, for the controlled "same explained model" comparison -----------
a4 = U.load_a4()
P3 = load_model("p3")
tm_pred = predict(P3, a4)
print("their p3 on 附件4:", tm_pred["prob"].shape)


_VOCAB = None
def _vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = {}
        with open(str(S.parent / "models" / "bert-tiny" / "vocab.txt")) as f:
            for k, line in enumerate(f):
                _VOCAB[k] = line.strip()
    return _VOCAB


def words_of(i):
    """Word token strings per position, decoded from 附件4's text_bert."""
    from src.q2_features import read_pickle
    from src import config as mycfg
    ids = []
    for p in sorted(Path(mycfg.ATT4_DIR).glob("*.pkl")):
        rec = read_pickle(p)
        tb = np.asarray(rec["text_bert"])
        ids.append(tb[0] if tb.ndim == 2 else tb[0])
    v = _vocab()
    return [v.get(int(t), "?") for t in ids[i]]


def pos_of_words(i, targets):
    """Map evidence word strings back to positions via the decoded token row."""
    ws = words_of(i)
    out = []
    for t in targets:
        t = str(t).strip()
        if not t:
            continue
        for j, w in enumerate(ws):
            if w == t and j not in out:
                out.append(j)
                break
    return out


def drop_effect_mine(i, positions):
    one = {k: v[i:i + 1] for k, v in arr.items()}
    base = float(np.mean([m.predict(one)["prob"][0, int(np.argmax(np.mean([mm.predict(one)["prob"] for mm in MINE], 0)[0]))] for m in MINE]))
    c = int(np.mean([m.predict(one)["prob"] for m in MINE], 0)[0].argmax())
    base = float(np.mean([m.predict(one)["prob"][0, c] for m in MINE]))
    mod = {k: v.copy() for k, v in one.items()}
    for j in positions:
        mod["T"][0, j] = 0.0
        mod["t_obs"][0, j] = 0.0
    after = float(np.mean([m.predict(mod)["prob"][0, c] for m in MINE]))
    return base - after, c


def drop_effect_p3(i, positions):
    """Same intervention, but on THEIR p3, using their tensor pipeline."""
    import copy
    s = a4[i]
    base = tm_pred["prob"][i]
    c = int(np.argmax(base))
    b0 = float(base[c])
    mod = copy.deepcopy(s)
    for j in positions:
        mod.text[j, :] = 0.0
        mod.mask["text"][j] = 0.0
    r = predict(P3, [mod])
    return b0 - float(r["prob"][0, c]), c


rows = []
for i in range(len(sp)):
    sid = str(sp.sample_id[i]).zfill(2)
    if sid not in tm.index:
        continue
    ev = tm.loc[sid, "首位文本证据词"]
    ev_all = tm.loc[sid, "text_evidence"]
    words = [ev]
    try:
        for d in json.loads(str(ev_all).replace("'", '"')):
            if isinstance(d, dict) and d.get("word"):
                words.append(d["word"])
    except Exception:
        pass
    words = list(dict.fromkeys([w for w in words if isinstance(w, str) and w.strip()]))[:5]
    pos_tm = pos_of_words(i, words)
    # mine: full occlusion sweep on the same explained model
    one = {k: v[i:i + 1] for k, v in arr.items()}
    c_mine = int(np.mean([m.predict(one)["prob"] for m in MINE], 0)[0].argmax())
    base_mine = float(np.mean([m.predict(one)["prob"][0, c_mine] for m in MINE]))
    wspan = [int(j) for j in np.flatnonzero(one["word_span"][0] > 0)]
    eff = {}
    for j in wspan:
        mod = {k: v.copy() for k, v in one.items()}
        mod["T"][0, j] = 0.0; mod["t_obs"][0, j] = 0.0
        eff[j] = base_mine - float(np.mean([m.predict(mod)["prob"][0, c_mine] for m in MINE]))
    pos_mine = sorted(eff, key=lambda j: -eff[j])[:max(1, len(pos_tm))]
    d_tm_mine, _ = drop_effect_mine(i, pos_tm) if pos_tm else (float("nan"), c_mine)
    d_my_mine, _ = drop_effect_mine(i, pos_mine)
    d_tm_p3, _ = drop_effect_p3(i, pos_tm) if pos_tm else (float("nan"), 0)
    d_my_p3, _ = drop_effect_p3(i, pos_mine)
    rows.append(dict(sid=sid, n_words=len(pos_tm),
                     teammate_words=";".join(words[:3]), teammate_pos=pos_tm,
                     drop_teammate_on_my_model=d_tm_mine, drop_mine_on_my_model=d_my_mine,
                     drop_teammate_on_p3=d_tm_p3, drop_mine_on_p3=d_my_p3))

df = pd.DataFrame(rows)
print("\n" + "=" * 100)
print(df[["sid", "n_words", "teammate_words", "drop_teammate_on_my_model",
          "drop_mine_on_my_model", "drop_teammate_on_p3", "drop_mine_on_p3"]].to_string(index=False))
print("=" * 100)
a = df.drop_teammate_on_my_model.mean(); b = df.drop_mine_on_my_model.mean()
c = df.drop_teammate_on_p3.mean(); d = df.drop_mine_on_p3.mean()
print(f"\nOn MY model  : teammate's evidence positions drop {a:.4f} | my occlusion-ranked drop {b:.4f}")
print(f"On THEIR p3  : teammate's evidence positions drop {c:.4f} | my occlusion-ranked drop {d:.4f}")
w1 = (df.drop_mine_on_my_model > df.drop_teammate_on_my_model).sum()
w2 = (df.drop_mine_on_p3 > df.drop_teammate_on_p3).sum()
print(f"\nPer-sample wins (mine > theirs): on my model {w1}/{len(df)}, on their p3 {w2}/{len(df)}")
df.to_csv("/tmp/eexp/exp17_evidence.csv", index=False)
print("\nwritten /tmp/eexp/exp17_evidence.csv")
