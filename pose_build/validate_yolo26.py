"""Validate raw YOLO26 ONNX/RKNN on images without modifying person records."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo26_pose import decode_yolo26


def letterbox(image):
    height, width = image.shape[:2]
    scale = min(640/height, 640/width)
    nh, nw = int(height*scale), int(width*scale)
    x, y = (640-nw)//2, (640-nh)//2
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[y:y+nh, x:x+nw] = cv2.resize(image, (nw, nh))
    return padded, scale, x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--reference", type=Path, help="Reference raw outputs from the ONNX run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = None
    if args.model.suffix == ".onnx":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        runtime = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
        infer = lambda image: runtime.run(None, {"images": image.transpose(2, 0, 1)[None].astype(np.float32)/255})
    else:
        from rknnlite.api import RKNNLite
        runtime = RKNNLite(verbose=False)
        if runtime.load_rknn(str(args.model)) != 0 or runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError("RKNN model initialization failed")
        infer = lambda image: runtime.inference(inputs=[image[None]])
    reports = []
    try:
        for path in args.images:
            original = cv2.imread(str(path))
            if original is None:
                raise ValueError(f"Cannot read {path}")
            padded, scale, pad_x, pad_y = letterbox(original)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            infer(rgb)
            started = time.perf_counter()
            for _ in range(args.runs):
                outputs = infer(rgb)
            elapsed = (time.perf_counter()-started)*1000/args.runs
            np.savez_compressed(args.output/f"{path.stem}-heads.npz", *outputs)
            detections = decode_yolo26(outputs, scale, pad_x, pad_y, original.shape[1], original.shape[0])
            report = {"image": path.name, "inference_ms": elapsed, "shapes": [list(o.shape) for o in outputs],
                      "detections": [{"box": d[:4], "score": d[4], "keypoints": d[5].tolist(),
                                      "confidence": d[6].tolist()} for d in detections]}
            if args.reference:
                with np.load(args.reference/f"{path.stem}-heads.npz") as saved:
                    ref = [saved[f"arr_{index}"] for index in range(9)]
                by_shape = {tuple(o.shape): o for o in ref}
                report["head_max_error"] = [float(np.max(np.abs(o-by_shape[tuple(o.shape)]))) for o in outputs]
                expected = decode_yolo26(ref, scale, pad_x, pad_y, original.shape[1], original.shape[0])
                if len(expected) != len(detections):
                    raise AssertionError(f"Detection counts differ: ONNX={len(expected)} RKNN={len(detections)}")
                errors = []
                for detection, reference in zip(detections, expected):
                    box_error = float(np.max(np.abs(np.asarray(detection[:4])-reference[:4])))
                    visible = reference[6] > 0.5
                    kp_error = float(np.max(np.abs(detection[5][visible]-reference[5][visible]))) if visible.any() else 0
                    errors.append({"box_max_error_px": box_error, "visible_kpt_max_error_px": kp_error})
                    if box_error > 10 or kp_error > 10:
                        raise AssertionError(f"Excessive conversion drift: {errors[-1]}")
                report["reference_errors"] = errors
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original, f"YOLO26 {detection[4]:.2f}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                for point, confidence in zip(detection[5], detection[6]):
                    if confidence > 0.3:
                        cv2.circle(original, tuple(map(int, point)), 4, (255, 0, 0), -1)
            cv2.imwrite(str(args.output/f"{path.stem}-annotated.jpg"), original)
            reports.append(report)
            print(json.dumps({k: v for k, v in report.items() if k != "detections"}, ensure_ascii=False),
                  f"people={len(detections)}", flush=True)
    finally:
        if args.model.suffix != ".onnx" and runtime is not None:
            runtime.release()
    (args.output/"report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
