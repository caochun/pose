#!/usr/bin/env python3
"""Validate an RK3576 OSNet ReID RKNN model on-device."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from rknnlite.api import RKNNLite


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    # The converter embeds ImageNet mean/std; feed a uint8 BGR crop.
    crop = cv2.resize(image, (128, 256), interpolation=cv2.INTER_LINEAR)
    batch = crop[None, ...]

    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(args.model)) != 0:
        raise SystemExit("load_rknn failed")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        raise SystemExit("init_runtime failed")
    first = np.asarray(rknn.inference(inputs=[batch])[0])
    second = np.asarray(rknn.inference(inputs=[batch])[0])
    print(f"output_shape={first.shape} output_dtype={first.dtype}")
    print(f"repeat_cosine={cosine(first, second):.6f}")

    for _ in range(5):
        rknn.inference(inputs=[batch])
    started = time.perf_counter()
    for _ in range(args.runs):
        rknn.inference(inputs=[batch])
    elapsed_ms = (time.perf_counter() - started) * 1000 / args.runs
    print(f"avg_ms={elapsed_ms:.3f} throughput_fps={1000 / elapsed_ms:.1f}")
    rknn.release()


if __name__ == "__main__":
    main()
