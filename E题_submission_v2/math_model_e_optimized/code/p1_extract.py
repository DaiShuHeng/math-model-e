"""Problem 1 media helpers; run p1_refine.py for the full pipeline.
Audio: custom 74-dimensional MFCC, spectral and pitch features.
Vision: 35 pixel-difference grid descriptors, not FACET or action-unit estimates.
These custom features are evaluated separately from the official attachment 2 interface."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

import imageio_ffmpeg  # noqa: E402

FFMPEG = None


def _ffmpeg() -> str:
    global FFMPEG
    if FFMPEG is None:
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


# ==========================================================================
# 解码
# ==========================================================================
def decode_audio(mp4) -> np.ndarray:
    """返回 16 kHz 单声道 float32 波形。"""
    sr = C.ACOUSTIC["sample_rate"]
    cmd = [_ffmpeg(), "-v", "error", "-i", str(mp4), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"音频解码失败 {mp4}: {p.stderr.decode('utf8','ignore')[:200]}")
    return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32) / 32768.0


def decode_frames(mp4, fps: float, size: Tuple[int, int]) -> np.ndarray:
    """以 fps 抽取灰度帧，缩放到 size=(W,H)，返回 (n, H, W) uint8。"""
    w, h = size
    cmd = [_ffmpeg(), "-v", "error", "-i", str(mp4),
           "-vf", f"fps={fps},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"视频解码失败 {mp4}: {p.stderr.decode('utf8','ignore')[:200]}")
    buf = np.frombuffer(p.stdout, dtype=np.uint8)
    n = len(buf) // (w * h)
    return buf[: n * w * h].reshape(n, h, w)


# ==========================================================================
# 音频特征：74 维
# ==========================================================================
def acoustic_features(y: np.ndarray) -> np.ndarray:
    import librosa

    cfg = C.ACOUSTIC
    sr, n_fft, hop = cfg["sample_rate"], cfg["n_fft"], cfg["hop_length"]
    y = np.asarray(y, dtype=np.float32)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    n_frames = 1 + len(y) // hop
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True))
    P = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    eps = 1e-10

    def fit(a: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(a)
        if a.shape[-1] < n_frames:
            a = np.pad(a, ((0, 0), (0, n_frames - a.shape[-1])), mode="edge")
        return a[..., :n_frames]

    # 1) MFCC 13 + Δ + ΔΔ
    mfcc = fit(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg["n_mfcc"], n_fft=n_fft, hop_length=hop))
    d1 = fit(librosa.feature.delta(mfcc))
    d2 = fit(librosa.feature.delta(mfcc, order=2))

    # 2) 对数能量 / Δ能量 / 过零率
    rms = fit(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True))
    log_e = np.log(rms + eps)
    d_e = fit(librosa.feature.delta(log_e))
    zcr = fit(librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop, center=True))

    # 3) log-F0 / ΔF0 / 浊音置信度
    try:
        f0 = librosa.yin(y, fmin=50.0, fmax=500.0, sr=sr, frame_length=1024, hop_length=hop)
        f0 = fit(f0)
    except Exception:
        f0 = np.zeros((1, n_frames), dtype=np.float32)
    log_f0 = np.log(np.maximum(f0, 1.0))
    d_f0 = fit(librosa.feature.delta(log_f0))
    d2_f0 = fit(librosa.feature.delta(log_f0, order=2))
    flat = fit(librosa.feature.spectral_flatness(S=S + eps))

    # 4) 谱形 7 维
    cen = fit(librosa.feature.spectral_centroid(S=S, sr=sr))
    bw = fit(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    ro85 = fit(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85))
    ro15 = fit(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.15))
    # 谱坡度：幅度谱对频率的线性回归斜率（librosa 1.0 已移除 spectral_slope，此处直接计算）
    fmean = freqs.mean()
    fden = ((freqs - fmean) ** 2).sum() + eps
    slope = ((freqs[:, None] - fmean) * (S - S.mean(axis=0, keepdims=True))).sum(axis=0) / fden
    slope = fit(slope[None, :])
    flux = np.zeros((1, n_frames), dtype=np.float32)
    dS = np.diff(S, axis=1, prepend=S[:, :1])
    flux[0, : dS.shape[1]] = np.sqrt((dS ** 2).mean(axis=0))[:n_frames]

    # 5) 谱对比度 7 子带
    try:
        contrast = fit(librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6))
    except Exception:
        contrast = np.zeros((7, n_frames), dtype=np.float32)

    # 6) 色度 12 维
    try:
        chroma = fit(librosa.feature.chroma_stft(S=S, sr=sr, n_chroma=12))
    except Exception:
        chroma = np.zeros((12, n_frames), dtype=np.float32)

    # 7) 谱峭度 / 谱熵 / 谱峰度因子
    Pn = P / (P.sum(axis=0, keepdims=True) + eps)
    spec_ent = -(Pn * np.log(Pn + eps)).sum(axis=0) / np.log(P.shape[0] + eps)
    mu = P.mean(axis=0, keepdims=True)
    sd = P.std(axis=0, keepdims=True)
    kurt = ((P - mu) ** 4).mean(axis=0) / (sd ** 4 + eps)
    crest = P.max(axis=0) / (mu[0] + eps)
    extra = fit(np.vstack([np.log1p(kurt), spec_ent, np.log1p(crest)]))

    feats = np.vstack([mfcc, d1, d2, log_e, d_e, zcr, log_f0, d_f0, d2_f0,
                       cen, bw, ro85, ro15, flat, slope, flux, contrast, chroma, extra])
    feats = np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    assert feats.shape[0] == C.DIM_AUDIO, f"音频特征维度 {feats.shape[0]} != {C.DIM_AUDIO}"
    return feats.T                                    # (n_frames, 74)


# ==========================================================================
# 视觉特征：35 维
# ==========================================================================
_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(path)
    return _CASCADE


def _detect_box(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """返回下采样坐标系下的 (x, y, w, h)。"""
    cas = _cascade()
    boxes = cas.detectMultiScale(gray, 1.1, C.FACE["min_neighbors"], minSize=(30, 30))
    if len(boxes) == 0:
        return None
    # 取面积最大者
    return tuple(max(boxes, key=lambda b: b[2] * b[3]))


def visual_features(mp4, fps: float = 10.0, return_details=False):
    """返回 (n_frames, 35) 面部像素差异网格，以及抽取帧的标称帧率。"""
    import cv2

    sc = C.FACE["detect_scale"]
    full = (320, 180)                                  # (W,H) 全分辨率用于裁剪
    frames = decode_frames(mp4, fps, full)
    n = frames.shape[0]
    if n == 0:
        result=(np.zeros((0, C.DIM_VISION), dtype=np.float32), fps)
        return (*result,np.zeros(0,bool)) if return_details else result

    h, w = frames.shape[1], frames.shape[2]
    fh, fw = C.FACE["face_size"]
    boxes: List[Optional[Tuple[int, int, int, int]]] = []
    for i in range(n):
        small = cv2.resize(frames[i], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        b = _detect_box(small)
        if b is not None:
            b = tuple(int(round(v / sc)) for v in b)
        boxes.append(b)

    good = [b for b in boxes if b is not None]
    if len(good) >= max(3, int(0.15 * n)):
        med = np.median(np.array(good, dtype=np.float64), axis=0)
        median_box = tuple(int(round(v)) for v in med)
    else:                                              # 检测失败过多 -> 取画面中央
        median_box = (int(w * 0.25), int(h * 0.05), int(w * 0.5), int(h * 0.85))

    crop = np.zeros((n, fh, fw), dtype=np.float32)
    for i in range(n):
        b = median_box
        x, y, bw, bh = b
        x = int(np.clip(x, 0, w - 2)); y = int(np.clip(y, 0, h - 2))
        bw = int(np.clip(bw, 8, w - x)); bh = int(np.clip(bh, 8, h - y))
        face = frames[i][y : y + bh, x : x + bw]
        crop[i] = cv2.resize(face, (fw, fh), interpolation=cv2.INTER_AREA).astype(np.float32)

    detected = np.array([b is not None for b in boxes])
    ref = np.median(crop[detected], axis=0) if detected.any() else np.median(crop, axis=0)
    dev = np.abs(crop - ref[None])                     # (n, fh, fw)

    r, c = C.FACE["grid_rows"], C.FACE["grid_cols"]
    rs, cs = fh // r, fw // c
    grid = dev[:, : r * rs, : c * cs].reshape(n, r, rs, c, cs).mean(axis=(2, 4))
    grid = grid.reshape(n, r * c)

    scale = np.percentile(grid, 90) if grid.size else 0.0
    if scale > 1e-6:
        grid = grid / scale
    grid = np.clip(grid, 0.0, 5.0)
    result=(grid.astype(np.float32), fps)
    return (*result,detected) if return_details else result


# ==========================================================================
# 对齐：任意帧率序列 -> 50 个等长时间位置
# ==========================================================================
