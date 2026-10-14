"""Export YOLO26 COCO pose raw heads for Rockchip's NPU.

Only backbone/neck/heads run on the NPU; decoding remains on the CPU.
Uses the same raw-head approach as YeWenxuan64/yolo26_ModelDeploy,
with a fixed 640x640 input and explicit output shapes for validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxslim
import torch
from ultralytics import YOLO


class RawPose(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model
        self.head = model.model[-1]
        identity = torch.nn.Identity()
        identity.f, identity.i = self.head.f, self.head.i
        model.model[-1] = identity

    def forward(self, image):
        features = self.net(image)
        # Official YOLO26 weights have end2end=False for the pretrained pose
        # checkpoint; after fuse(), the usable branch is exposed as one2many.
        heads = self.head.one2one
        if heads["box_head"] is None:
            heads = self.head.one2many
        boxes = [heads["box_head"][i](features[i]) for i in range(self.head.nl)]
        scores = [heads["cls_head"][i](features[i]).sigmoid() for i in range(self.head.nl)]
        points = [heads["kpts_head"][i](heads["pose_head"][i](features[i]))
                  for i in range(self.head.nl)]
        return tuple(boxes + scores + points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Official yolo26n-pose.pt weights")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = YOLO(args.weights).model.eval().float().fuse(verbose=False)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.modules():
        if hasattr(layer, "forward_split"):
            layer.forward = layer.forward_split
    wrapper = RawPose(model).eval()
    if wrapper.head.reg_max != 1 or tuple(wrapper.head.kpt_shape) != (17, 3):
        raise ValueError("Expected a YOLO26 COCO pose head with direct box regression")
    names = [f"{kind}_p{i}" for kind in ("box", "score", "kpts") for i in (3, 4, 5)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, torch.zeros(1, 3, 640, 640), str(args.output),
                      input_names=["images"], output_names=names, opset_version=13,
                      dynamic_axes=None, dynamo=False)
    graph = onnxslim.slim(onnx.load(args.output))
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, args.output)
    for output in graph.graph.output:
        print(output.name, [dimension.dim_value for dimension in output.type.tensor_type.shape.dim])
    print(f"Exported {args.output}", flush=True)


if __name__ == "__main__":
    main()
