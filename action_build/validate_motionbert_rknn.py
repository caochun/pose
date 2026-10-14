"""Benchmark RKNN on a recorded skeleton sequence against ONNX logits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from rknnlite.api import RKNNLite

from motionbert_input import predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path, help="Prepared [1,2,32,17,3] .npy input")
    parser.add_argument("reference", type=Path, help="ONNX logits .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    x = np.load(args.input).astype(np.float32)
    if x.shape != (1, 2, 32, 17, 3):
        raise ValueError(f"Incorrect input shape {x.shape}")
    expected = np.load(args.reference).reshape(-1)
    names = json.loads(Path(__file__).with_name("ntu60_labels.json").read_text())
    rknn = RKNNLite(verbose=False)
    try:
        if rknn.load_rknn(str(args.model)) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError("RKNN initialization failed")
        def infer():
            # This is a skeleton tensor, not an NHWC image.
            return np.asarray(rknn.inference(inputs=[x[0]], data_format=["nchw"])[0]).reshape(-1)
        actual = infer()
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise AssertionError("RKNN output shape/values are invalid")
        started = time.perf_counter()
        for _ in range(args.runs):
            actual = infer()
        elapsed_ms = (time.perf_counter() - started) * 1000 / args.runs
        report = {
            "model": args.model.name, "model_bytes": args.model.stat().st_size,
            "inference_ms": elapsed_ms,
            "logit_max_error": float(np.max(np.abs(actual - expected))),
            "top1_matches": bool(actual.argmax() == expected.argmax()),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "candidate_actions": predictions(actual, names),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not report["top1_matches"] or report["logit_max_error"] > 0.5:
            raise AssertionError("Excessive RKNN conversion drift")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
