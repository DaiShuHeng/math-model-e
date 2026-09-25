"""Reproduce clean teacher validation and export TRAIN-only soft targets.
No TEST or attachment 3/4 inference. Teacher checkpoints are existing p2 runs.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solution'))
sys.path.insert(0, str(ROOT / 'math_model_e_optimized' / 'code'))
from src.data_adapter import load_training_splits
from src.distillation import split_fingerprint
import config as C
import io_utils as U
import models as M
import trainer as T
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(args.threads)
    # Refit normalization on TRAIN instead of trusting a path-based legacy cache.
    raw = U.load_a2_all(normalize=False, splits=('train', 'valid'))
    stats = U.compute_stats(raw['train'])
    data = {s: U.apply_stats(x, stats) for s, x in raw.items()}
    del raw
    U._a2_raw.cache_clear()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    totals = {s: [] for s in data}
    provenance = []
    for seed in (2026, 2027, 2028):
        path = C.MODEL_DIR / f'p2_{seed}' / 'model.pt'
        ck = torch.load(path, map_location='cpu', weights_only=True)
        if ck['interface'] != 'bert_direct_content_mask_v2' or ck['model_cfg'] != C.model_cfg_dict():
            raise ValueError('Unexpected teacher architecture/interface')
        model = M.RobustModel().to(device)
        model.load_state_dict(ck['state'], strict=True)
        provenance.append({'path': str(path.relative_to(ROOT)),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for split, samples in data.items():
            dl = DataLoader(T.MoseiDataset(samples), batch_size=128, collate_fn=T.collate)
            result = T.predict(model, dl, device)
            totals[split].append(result)
            print(seed, split, result['metrics'], flush=True)
        del model
    reports = {}
    avg = {}
    for split, runs in totals.items():
        prob = np.mean([r['prob'] for r in runs], axis=0)
        reg = np.mean([r['p_reg'] for r in runs], axis=0)
        avg[split] = (prob, reg)
        reports[split] = T.compute_metrics(runs[0]['y_cls'], prob.argmax(-1), runs[0]['y_reg'], reg)
    # This is a reproduction check, not hyperparameter tuning.
    if abs(reports['valid']['f1_macro'] - 0.6347761639791254) > 0.002:
        raise ValueError(f'Teacher validation differs from recorded reference: {reports}')
    sd = load_training_splits()['train']
    if [s.sid for s in data['train']] != list(sd.sample_id):
        raise ValueError('Teacher/student sample identity mismatch')
    if not np.array_equal([s.label_cls for s in data['train']], sd.y_cls):
        raise ValueError('Teacher/student label convention mismatch')
    args.out.mkdir(parents=True)
    prob, reg = avg['train']
    # log(mean probabilities) supplies logits whose softmax exactly reproduces
    # the existing probability ensemble, not a new logit-averaged teacher.
    np.savez_compressed(args.out/'teacher_train.npz', split='train',
                        fingerprint=split_fingerprint(sd), sample_id=np.asarray(sd.sample_id).astype(str),
                        logits=np.log(prob.clip(1e-8, 1)), reg=reg)
    vp, vr = avg['valid']
    np.savez_compressed(args.out/'teacher_valid_predictions.npz',
                        sample_id=np.asarray([s.sid for s in data['valid']]),
                        prob=vp, reg=vr, y_cls=totals['valid'][0]['y_cls'], y_reg=totals['valid'][0]['y_reg'])
    report = {'teacher': 'p2 probability ensemble, uncalibrated', 'checkpoints': provenance,
              'normalizer_fit': 'train only, freshly recomputed', 'metrics': reports,
              'scope': 'TRAIN soft targets; VALID reproduction only; no TEST/A3/A4 evaluation'}
    (args.out/'teacher_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
