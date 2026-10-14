"""Convert a raw-head YOLO26 ONNX using RKNN Toolkit2 2.5.0."""
from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", help="RGB uint8 calibration images; omit for FP16")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        steps = [
            ("config", lambda: rknn.config(target_platform="rk3576", optimization_level=3,
                                          mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                                          disable_rules=['fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa'])),
            ("load", lambda: rknn.load_onnx(model=str(args.onnx))),
            ("build", lambda: rknn.build(do_quantization=bool(args.dataset), dataset=args.dataset)),
            ("export", lambda: rknn.export_rknn(str(args.output))),
        ]
        for name, operation in steps:
            print(f"{name}: {args.output}", flush=True)
            result = operation()
            if result != 0:
                raise RuntimeError(f"{name} failed: {result}")
    finally:
        rknn.release()
    print(f"Exported {args.output}: {args.output.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
