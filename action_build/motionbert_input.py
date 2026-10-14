"""COCO poses -> the official MotionBERT NTU60 action input convention."""
from __future__ import annotations

import numpy as np


def coco2h36m(x):
    """Map (..., 17, channels) COCO joints to H36M, including virtual joints."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-2] != 17:
        raise ValueError("Expected COCO 17 joints")
    y = np.zeros_like(x)
    y[..., 0, :] = (x[..., 11, :] + x[..., 12, :]) * 0.5
    for dst, src in ((1, 12), (2, 14), (3, 16), (4, 11), (5, 13), (6, 15),
                     (9, 0), (11, 5), (12, 7), (13, 9), (14, 6), (15, 8), (16, 10)):
        y[..., dst, :] = x[..., src, :]
    y[..., 8, :] = (x[..., 5, :] + x[..., 6, :]) * 0.5
    y[..., 7, :] = (y[..., 0, :] + y[..., 8, :]) * 0.5
    y[..., 10, :] = (x[..., 1, :] + x[..., 2, :]) * 0.5
    return y


def prepare_input(keypoints, confidence, frame_size, frames=32):
    """Return [1, 2, T, 17, 3], matching official NTU60 test preprocessing.

    The official action loader maps coordinates but keeps confidence channels
    in COCO order. Preserve that convention for these pretrained weights.
    It also pads a single-person clip with a zero second person BEFORE scaling.
    """
    xy = np.asarray(keypoints, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1:] != (17, 2) or conf.shape != xy.shape[:2]:
        raise ValueError("Expected T x 17 x 2 coordinates and T x 17 confidence")
    if len(xy) < 2 or not np.isfinite(xy).all() or not np.isfinite(conf).all():
        raise ValueError("Need at least two finite poses")
    width, height = frame_size
    if width <= 0 or height <= 0 or not 1 <= frames <= 243:
        raise ValueError("Invalid image dimensions or sequence length")
    xy = coco2h36m(xy[None] / max(width, height) * 2 - 1)
    motion = np.concatenate((xy, np.clip(conf[None, ..., None], 0, 1)), axis=-1)
    indices = np.linspace(0, motion.shape[1], num=frames, endpoint=False, dtype=int)
    motion = motion[:, indices]
    motion = np.concatenate((motion, np.zeros_like(motion)), axis=0)
    coords = motion[motion[..., 2] != 0, :2]
    if len(coords) < 4:
        raise ValueError("Not enough valid joints")
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    scale = float((hi - lo).max() * 2)  # scale_range_test = [2, 2]
    if scale <= 0:
        raise ValueError("Degenerate skeleton clip")
    origin = (lo + hi - scale) / 2
    motion[..., :2] = ((motion[..., :2] - origin) / scale - 0.5) * 2
    return np.clip(motion[None], -1, 1).astype(np.float32)


def predictions(logits, names, topk=5):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if len(logits) != len(names) or not np.isfinite(logits).all():
        raise ValueError("Invalid action logits")
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return [{"id": int(i), "label": names[i], "score": float(probs[i])}
            for i in np.argsort(probs)[-topk:][::-1]]
