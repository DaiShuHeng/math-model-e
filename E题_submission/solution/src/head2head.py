"""Head-to-head: my Q2 champion ensemble vs teammate's p2 ensemble on IDENTICAL
corrupted VALID copies (adjudicated 附件3 protocol, per-sample seed 0+index).

Conditions: clean | pool (P_FILE=0.9 + empirical rate pool) | r20 | r40.
Damage is computed ONCE per condition/sample in raw id/array space, then fed
through each pipeline's own standardisation/interface:
  - mine:  damaged ids -> BERT-Tiny; damaged A/V -> my train-fit standardizer.
  - theirs: damaged text_bert re-encoded through frozen bert-base (their
    附件3 interface), damaged A/V -> their train-fit stats; ensemble of 3
    seeds + their valid-frozen logit bias [-0.2, 0, 0].
Also reports cross-pipeline agreement and a 50/50 probability ensemble.

Run:
  cd solution && CUDA_VISIBLE_DEVICES=0 python -m src.head2head
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SOL = Path(__file__).resolve().parents[1]
TM = _E_ROOT / "math_model_e_optimized"
os.environ.setdefault("MATH_E_DATA", str(_E_ROOT / "E题" / "E题数据"))
os.environ.setdefault("MATH_E_BERT", str(_E_ROOT / "models" / "bert-base-uncased"))
sys.path.insert(0, str(TM / "code"))

from src import config as mycfg                      # noqa: E402
from src.augment import WordLevelTAV                 # noqa: E402
from src.data_adapter import Standardizer, load_splits  # noqa: E402
from src.metrics import cls_metrics, head_agreement, reg_metrics  # noqa: E402
from src.models import Q2Model                       # noqa: E402

import io_utils as U                                 # noqa: E402  (teammate's)
import inference as TI                               # noqa: E402
from text_encoder import encode_text_bert_batch      # noqa: E402
_E_ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = ("clean", "pool", "r20", "r40")
MY_CKPTS = sorted((SOL / "weights" / "q2_ft_cw").glob("s*/best.pt"))          # v1 champion (Tiny, ft)
CHAMP_CKPTS = sorted((SOL / "weights" / "q2_base_cw").glob("s*/best.pt"))      # v2 champion (base, frozen)
THEIR_BIAS = np.array([-0.2, 0.0, 0.0])              # models/decision_calibration.json p2


def make_aug(cond: str, j: int):
    if cond == "clean":
        return None
    if cond == "pool":
        return WordLevelTAV(p_file=mycfg.P_FILE, seed=0 + j)   # empirical rate pool
    r = 0.2 if cond == "r20" else 0.4
    return WordLevelTAV(p_file=1.0, rate_pool=[r], seed=0 + j)


@torch.no_grad()
def my_predict(models, batch, device):
    probs, regs, gates = [], [], []
    for m in models:
        out = m(batch)
        probs.append(torch.softmax(out["logits"].float(), dim=-1))
        regs.append(out["reg"].float())
        gates.append(out["gate"].float())
    prob = torch.stack(probs).mean(0).cpu().numpy()
    reg = torch.stack(regs).mean(0).cpu().numpy()
    gate = torch.stack(gates).mean(0).mean(0).cpu().numpy()
    return {"prob": prob, "p_cls": prob.argmax(1), "p_reg": reg, "gate_mean": gate}


def their_predict(net, samples, bias):
    res = TI.predict(net, samples)                    # cpu, no augmentation
    logp = np.log(res["prob"].clip(1e-9)) + bias
    prob = np.exp(logp - logp.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    return {"prob": prob, "p_cls": prob.argmax(1), "p_reg": res["p_reg"],
            "alpha_mean": res["alpha"].mean(0)}


def score(y_c, p_c, y_r, p_r):
    m = cls_metrics(y_c, p_c)
    m.update(reg_metrics(y_r, p_r))
    m["head_agreement"] = head_agreement(p_c, p_r)
    return m


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- my side ----
    splits = load_splits()
    std = Standardizer.fit(splits["train"])
    def load_ensemble(ckpts):
        out = []
        for c in ckpts:
            ck = torch.load(c, map_location="cpu", weights_only=False)
            m = Q2Model(bert_dir=ck["args"].get("bert_dir", str(mycfg.BERT_DIR)),
                        freeze_text=bool(ck["args"].get("freeze_text", 1))).to(device).eval()
            m.load_state_dict(ck["state_dict"])
            out.append(m)
        return out

    my_models = load_ensemble(MY_CKPTS)
    champ_models = load_ensemble(CHAMP_CKPTS)
    print(f"my v1 ensemble: {[c.parent.name for c in MY_CKPTS]}", flush=True)
    print(f"champion ensemble: {[c.parent.name for c in CHAMP_CKPTS]}", flush=True)

    # ---- their side (raw valid samples keep meta['text_bert'] (3,50)) ----
    split_name = os.environ.get("H2H_SPLIT", "valid")
    raw_valid = U.load_a2_all(normalize=False, splits=(split_name,))[split_name]
    stats = U.load_stats(U.STATS_PATH)
    their_net = TI.load_model("p2")
    y_c = np.array([s.label_cls for s in raw_valid])
    y_r = np.array([s.label_reg for s in raw_valid])
    n = len(raw_valid)

    # sanity: my cached ids must match their text_bert channel 0
    for j in (0, 17, 400, n - 1):
        assert np.array_equal(raw_valid[j].meta["text_bert"][0], splits[split_name].ids[j]), j
    print("id interface check: OK", flush=True)

    results = {}
    for cond in CONDITIONS:
        tb_d, aud_d, vis_d = [], [], []
        n_damaged = 0
        for j, s in enumerate(raw_valid):
            tb = s.meta["text_bert"].copy()
            audio = s.audio.copy()
            vision = s.vision.copy()
            aug = make_aug(cond, j)
            if aug is not None:
                dmg = aug(tb[0], tb[1], audio, vision)
                n_damaged += int(dmg.any())
            tb_d.append(tb)
            aud_d.append(audio)
            vis_d.append(vision)
        print(f"[{cond}] damaged {n_damaged}/{n}", flush=True)

        # ---- their features: re-encode damaged text_bert through bert-base ----
        text_enc = encode_text_bert_batch([tb.T for tb in tb_d])
        their_samples = []
        for j, s in enumerate(raw_valid):
            sup = U.token_support(tb_d[j])
            feats = {"text": text_enc[j], "audio": aud_d[j], "vision": vis_d[j]}
            mask = {m: U.valid_mask(feats[m]) * sup for m in U.C.MODALITIES}
            their_samples.append(U.Sample(
                s.sid, feats["text"], feats["audio"], feats["vision"], mask,
                {m: sup.copy() for m in U.C.MODALITIES},
                label_cls=s.label_cls, label_reg=s.label_reg,
                meta={"text_bert": tb_d[j]}))
        their_samples = U.apply_stats(their_samples, stats)
        th = their_predict(their_net, their_samples, THEIR_BIAS)

        # ---- my features: same damaged arrays through my standardizer ----
        A = np.stack(aud_d)
        V = np.stack(vis_d)
        a_obs = ~np.isclose(A, 0).all(axis=2)
        v_obs = ~np.isclose(V, 0).all(axis=2)
        A_z = std.transform_audio(A, a_obs)
        V_z = std.transform_vision(V, v_obs)
        batch = {
            "ids": torch.from_numpy(np.stack([tb[0] for tb in tb_d])).to(device),
            "attn": torch.from_numpy(np.stack([tb[1] for tb in tb_d]).astype(np.int64)).to(device),
            "audio": torch.from_numpy(A_z.astype(np.float32)).to(device),
            "vision": torch.from_numpy(V_z.astype(np.float32)).to(device),
            "audio_obs": torch.from_numpy(a_obs.astype(np.float32)).to(device),
            "vision_obs": torch.from_numpy(v_obs.astype(np.float32)).to(device),
        }
        me = my_predict(my_models, batch, device)
        ch = my_predict(champ_models, batch, device)

        ens_prob = 0.5 * ch["prob"] + 0.5 * th["prob"]
        row = {
            "mine": score(y_c, me["p_cls"], y_r, me["p_reg"]),
            "champ": score(y_c, ch["p_cls"], y_r, ch["p_reg"]),
            "theirs": score(y_c, th["p_cls"], y_r, th["p_reg"]),
            "ensemble50": score(y_c, ens_prob.argmax(1), y_r, 0.5 * (ch["p_reg"] + th["p_reg"])),
            "agree_cls": float((me["p_cls"] == th["p_cls"]).mean()),
            "agree_champ_theirs": float((ch["p_cls"] == th["p_cls"]).mean()),
            "corr_reg": float(np.corrcoef(me["p_reg"], th["p_reg"])[0, 1]),
            "my_gate": me["gate_mean"].round(4).tolist(),
            "champ_gate": ch["gate_mean"].round(4).tolist(),
            "their_alpha": th["alpha_mean"].round(4).tolist(),
        }
        results[cond] = row
        f = lambda m: f"Acc {m['Accuracy']:.4f} mF1 {m['MacroF1']:.4f} MAE {m['MAE']:.4f} r {m['Pearson']:.4f}"
        print(f"[{cond:>5}] mine   {f(row['mine'])}", flush=True)
        print(f"[{cond:>5}] champ  {f(row['champ'])}", flush=True)
        print(f"[{cond:>5}] theirs {f(row['theirs'])}", flush=True)
        print(f"[{cond:>5}] ens50  {f(row['ensemble50'])} agree {row['agree_cls']:.3f}", flush=True)

    out = SOL / "logs" / ("head2head.json" if split_name == "valid" else f"head2head_{split_name}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
