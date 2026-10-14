"""Attempt FP16 MotionBERT compilation for RK3576; no image normalization."""
from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        for name, operation in (
            ("config", lambda: rknn.config(target_platform="rk3576", optimization_level=3,
                                          disable_rules=["fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa",
                                                         "squeeze_nd_mul",
                                                         "swap_reshape_exsoftmax13"])),
            ("load", lambda: rknn.load_onnx(model=str(args.onnx))),
            ("build", lambda: rknn.build(do_quantization=False)),
            ("export", lambda: rknn.export_rknn(str(args.output))),
        ):
            print(f"{name}: {args.output}", flush=True)
            result = operation()
            if result != 0:
                raise RuntimeError(f"{name} failed: {result}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
