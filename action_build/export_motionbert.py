"""Export an official MotionBERT NTU60 classification checkpoint to fixed ONNX."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

import numpy as np
import onnx
import onnxruntime as ort
import torch


def temporal_attention_4d(self, q, k, v, seqlen=8):
    """Equivalent attention with joint/head axes merged, avoiding rank-5 MatMul."""
    bt, _, joints, channels = q.shape
    def reshape(value):
        return value.reshape(-1, seqlen, self.num_heads, joints, channels).permute(
            0, 2, 3, 1, 4).reshape(-1, self.num_heads * joints, seqlen, channels)
    qt, kt, vt = reshape(q), reshape(k), reshape(v)
    weights = self.attn_drop(((qt @ kt.transpose(-2, -1)) * self.scale).softmax(dim=-1))
    x = weights @ vt
    return x.reshape(-1, self.num_heads, joints, seqlen, channels).permute(
        0, 3, 2, 1, 4).reshape(bt, joints, channels * self.num_heads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Official MotionBERT checkout")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--flat-input", action="store_true", help="Four-dimensional input for RKNN")
    parser.add_argument("--rknn-friendly", action="store_true", help="Equivalent rank-4 temporal attention")
    parser.add_argument("--sample", type=Path, help="Prepared action input .npy")
    args = parser.parse_args()
    args.flat_input = args.flat_input or args.rknn_friendly
    if not 1 <= args.frames <= 243:
        parser.error("frames must be 1..243")
    sys.path.insert(0, str(args.source.resolve()))
    from lib.model.DSTformer import DSTformer
    from lib.model.model_action import ActionNet

    torch.set_num_threads(4)
    backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
                         depth=5, num_heads=8, mlp_ratio=2, maxlen=243, num_joints=17,
                         norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    model = ActionNet(backbone, dim_rep=512, num_classes=60,
                      dropout_ratio=0.5, num_joints=17).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    if args.flat_input:
        class FlatInput(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, skeletons):
                return self.net(skeletons.unsqueeze(0))
        model = FlatInput(model).eval()
    sample = (np.load(args.sample) if args.sample else
              np.random.default_rng(42).uniform(-1, 1, (1, 2, args.frames, 17, 3)).astype(np.float32))
    if sample.shape != (1, 2, args.frames, 17, 3):
        raise ValueError(f"Incorrect sample shape: {sample.shape}")
    if args.flat_input:
        sample = sample[0]
    x = torch.from_numpy(sample)
    with torch.inference_mode():
        expected = model(x).numpy()
        if args.rknn_friendly:
            for module in model.modules():
                if getattr(module, "mode", None) == "temporal":
                    module.forward_temporal = MethodType(temporal_attention_4d, module)
            np.testing.assert_allclose(model(x).numpy(), expected, rtol=1e-4, atol=1e-4)
        started = time.perf_counter()
        model(x)
        torch_ms = (time.perf_counter() - started) * 1000
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, x, str(args.output), input_names=["skeletons"],
                          output_names=["logits"], opset_version=17, dynamo=False,
                          external_data=False)
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"skeletons": sample})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    started = time.perf_counter()
    session.run(None, {"skeletons": sample})
    ort_ms = (time.perf_counter() - started) * 1000
    with args.checkpoint.open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    report = {
        "source": "https://github.com/Walter0807/MotionBERT",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": "walterzhu/MotionBERT/checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin",
        "checkpoint_sha256": checkpoint_sha256,
        "frames": args.frames, "input_shape": list(sample.shape),
        "rknn_friendly": args.rknn_friendly,
        "validation_input": "recorded COCO sequence" if args.sample else "synthetic sequence",
        "torch_ms": torch_ms, "onnx_ms": ort_ms,
        "logit_max_error": float(np.max(np.abs(expected - actual))),
        "model_bytes": args.output.stat().st_size,
        "status": "experimental; NTU60 categories, no idle/unknown class",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
