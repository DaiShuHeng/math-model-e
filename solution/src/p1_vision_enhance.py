"""Q1 vision-branch enhancement: rerun the face-detection stage with YuNet.

Problem: the teammate's p1 pipeline uses OpenCV Haar cascade; on 39/100 附件1
videos it detects (almost) nothing, so mask_vision collapses to 0 and the whole
vision modality is declared unobserved. This script re-runs ONLY the vision
stage with the stronger YuNet detector (face_detection_yunet_2023mar.onnx,
opencv_zoo), replicating their exact downstream logic (largest-box per frame,
median reference box when >= max(3, 15%) frames detected else center crop,
7x5 deviation grid, 90th-pct normalize, clip [0,5], same 10 fps / 320x180
gray decode) and writes v3 artifacts NEXT TO the untouched v2 ones:

  solution/data/p1_features_v3/<sid>.npz   (vision, mask_vision, meta updated)
  solution/data/p1_summary_v3.csv          (vision_valid / face_detection_rate)
  solution/logs/p1_vision_v3_report.json   (before/after aggregates)

If YuNet's hit rate on a video is below the v2 Haar rate, Haar is re-run on
that video and the two detections are unioned (documented fallback).

Run (CPU only, ~5 min):
  MATH_E_DATA=/home/daishuheng/math_competition/E题/E题数据 \
    python -m src.p1_vision_enhance
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_E_ROOT = Path(__file__).resolve().parents[2]
SOL = Path(__file__).resolve().parents[1]
TM = _E_ROOT / "math_model_e_optimized"
os.environ.setdefault("MATH_E_DATA", str(_E_ROOT / "E题" / "E题数据"))
sys.path.insert(0, str(TM / "code"))

import config as C                     # noqa: E402  (teammate's config)
from p1_extract import decode_frames, _cascade, _detect_box  # noqa: E402

YUNET = _E_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
FPS = 10.0
SIZE = (320, 180)                      # (W, H) — identical to their protocol
OUT_DIR = SOL / "data" / "p1_features_v3"
SCORE = 0.6

_det = None


def _yunet():
    global _det
    if _det is None:
        _det = cv2.FaceDetectorYN.create(str(YUNET), "", SIZE, SCORE, 0.3, 5000)
    return _det


def detect_frames(frames: np.ndarray, v2_rate: float):
    """Per-frame largest box with YuNet; Haar union only if YuNet rate < v2's."""
    det = _yunet()
    n = len(frames)
    boxes = [None] * n
    hits = 0
    for i in range(n):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        ok, faces = det.detect(bgr)
        if ok and faces is not None and len(faces):
            # row: x, y, w, h, + 11 landmarks/score; keep largest area
            boxes[i] = tuple(int(round(v)) for v in max(faces[:, :4], key=lambda b: b[2] * b[3]))
            hits += 1
    if hits / max(n, 1) < v2_rate:                 # YuNet worse than v2 Haar -> union
        cas = _cascade()
        sc = C.FACE["detect_scale"]
        for i in range(n):
            if boxes[i] is not None:
                continue
            small = cv2.resize(frames[i], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            b = _detect_box(small)
            if b is not None:
                boxes[i] = tuple(int(round(v / sc)) for v in b)
                hits += 1
    return boxes, hits, n


def grid_from_boxes(frames: np.ndarray, boxes) -> np.ndarray:
    """Replicates their visual_features downstream given per-frame boxes."""
    h, w = frames.shape[1], frames.shape[2]
    fh, fw = C.FACE["face_size"]
    n = len(frames)
    good = [b for b in boxes if b is not None]
    if len(good) >= max(3, int(0.15 * n)):
        med = np.median(np.array(good, dtype=np.float64), axis=0)
        median_box = tuple(int(round(v)) for v in med)
    else:
        median_box = (int(w * 0.25), int(h * 0.05), int(w * 0.5), int(h * 0.85))
    crop = np.zeros((n, fh, fw), dtype=np.float32)
    for i in range(n):
        x, y, bw, bh = median_box
        x = int(np.clip(x, 0, w - 2)); y = int(np.clip(y, 0, h - 2))
        bw = int(np.clip(bw, 8, w - x)); bh = int(np.clip(bh, 8, h - y))
        face = frames[i][y: y + bh, x: x + bw]
        crop[i] = cv2.resize(face, (fw, fh), interpolation=cv2.INTER_AREA).astype(np.float32)
    detected = np.array([b is not None for b in boxes])
    ref = np.median(crop[detected], axis=0) if detected.any() else np.median(crop, axis=0)
    dev = np.abs(crop - ref[None])
    r, c = C.FACE["grid_rows"], C.FACE["grid_cols"]
    rs, cs = fh // r, fw // c
    grid = dev[:, : r * rs, : c * cs].reshape(n, r, rs, c, cs).mean(axis=(2, 4))
    grid = grid.reshape(n, r * c)
    scale = np.percentile(grid, 90) if grid.size else 0.0
    if scale > 1e-6:
        grid = grid / scale
    return np.clip(grid, 0.0, 5.0).astype(np.float32), detected


def pool_vision(grid: np.ndarray, detected: np.ndarray, edges: np.ndarray):
    """Their pool_frames rule: bin value = mean over DETECTED frames in the bin,
    mask = 1 iff at least one detected frame falls in the bin (10 fps frames)."""
    times = (np.arange(len(detected)) + 0.5) / FPS
    out = np.zeros((len(edges) - 1, grid.shape[1]), np.float32)
    mask = np.zeros(len(out), np.float32)
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        pick = (times >= a) & (times < b) & detected
        if pick.any():
            out[j] = grid[pick].mean(0)
            mask[j] = 1
    return out, mask


def process(job):
    sid, mp4, v2_rate = job
    frames = decode_frames(str(mp4), FPS, SIZE)
    if len(frames) == 0:
        return sid, dict(rate=0.0, n=0)
    boxes, hits, n = detect_frames(frames, v2_rate)
    grid, detected = grid_from_boxes(frames, boxes)
    return sid, dict(rate=hits / n, n=n, detected=detected, grid=grid)


def main():
    summ = pd.read_csv(TM / "data" / "p1_summary_v2.csv", dtype={"video_id": str, "clip_id": str})
    rate_map = dict(zip(summ.sample_id, summ.face_detection_rate))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for _, row in summ.iterrows():
        sid_raw = f"{row.video_id}__{row.clip_id}"
        mp4 = C.A1_DIR / row.video_id / f"{row.clip_id}.mp4"
        jobs.append((sid_raw, mp4, float(rate_map[row.sample_id])))
    print(f"{len(jobs)} videos", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process, j): j[0] for j in jobs}
        for k, fut in enumerate(as_completed(futs)):
            sid = futs[fut]
            try:
                results[sid] = fut.result()[1]
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {sid}: {e}", flush=True)
            print(f"{len(results)}/{len(jobs)}", flush=True)

    rows = []
    v2_novision = 0
    v3_novision = 0
    for _, row in summ.iterrows():
        sid_raw = f"{row.video_id}__{row.clip_id}"
        v2 = np.load(TM / "data" / "p1_features" / f"{sid_raw}.npz", allow_pickle=False)
        r = results.get(sid_raw)
        if r is None:  # keep v2 unchanged on failure
            vision = v2["vision"]; mask_v = v2["mask_vision"]; rate = float(rate_map[row.sample_id])
        else:
            edges = v2["time_edges"].astype(np.float64)
            if r["n"] == 0:
                vision = v2["vision"]; mask_v = v2["mask_vision"]; rate = 0.0
            else:
                vision, mask_v = pool_vision(r["grid"], r["detected"], edges); rate = r["rate"]
        meta = v2["meta"].copy()
        meta[-1] = rate
        np.savez_compressed(
            OUT_DIR / f"{sid_raw}.npz",
            text=v2["text"], audio=v2["audio"], vision=vision,
            mask_text=v2["mask_text"], mask_audio=v2["mask_audio"], mask_vision=mask_v,
            time_edges=v2["time_edges"], meta=meta, schema_version=v2["schema_version"],
        )
        out = dict(row)
        out["vision_valid"] = int(mask_v.sum())
        out["face_detection_rate"] = rate
        out["vision_changed"] = bool(r is not None and r["n"] > 0)
        rows.append(out)
        v2_novision += int(row.vision_valid == 0)
        v3_novision += int(mask_v.sum() == 0)

    df = pd.DataFrame(rows)
    df.to_csv(SOL / "data" / "p1_summary_v3.csv", index=False, encoding="utf-8-sig")
    report = {
        "v2_vision_valid_zero": v2_novision,
        "v3_vision_valid_zero": v3_novision,
        "v2_mean_face_rate": float(summ.face_detection_rate.mean()),
        "v3_mean_face_rate": float(df.face_detection_rate.mean()),
        "v2_mean_vision_valid_bins": float(summ.vision_valid.mean()),
        "v3_mean_vision_valid_bins": float(df.vision_valid.mean()),
    }
    (SOL / "logs").mkdir(exist_ok=True)
    (SOL / "logs" / "p1_vision_v3_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
