"""Decode RKNN YOLO26 pose raw heads to the monitor's detection format."""
from __future__ import annotations

import numpy as np


def decode_yolo26(outputs, scale, pad_x, pad_y, orig_w, orig_h,
                  conf_threshold=0.60, input_size=640):
    """Decode direct ltrb boxes and anchor-relative COCO keypoints.

    The export emits nine NCHW tensors: three box heads (4 channels),
    three sigmoid score heads (1), and three keypoint heads (51).
    YOLO26 uses neither YOLOv8's DFL nor its doubled keypoint offsets.
    """
    heads = {}
    for output in outputs:
        array = np.asarray(output)
        if array.ndim != 4 or array.shape[0] != 1:
            raise ValueError(f"Unexpected YOLO26 tensor shape: {array.shape}")
        key = tuple(array.shape[1:])
        if key in heads:
            raise ValueError(f"Duplicate YOLO26 tensor shape: {array.shape}")
        heads[key] = array[0]
    if len(heads) != 9:
        raise ValueError(f"Expected nine YOLO26 raw heads, got {len(heads)}")
    result = []
    for stride in (8, 16, 32):
        size = input_size // stride
        try:
            boxes = heads[(4, size, size)].reshape(4, -1)
            scores = heads[(1, size, size)].reshape(-1)
            points = heads[(51, size, size)].reshape(17, 3, -1)
        except KeyError as exc:
            raise ValueError(f"Missing YOLO26 head for stride {stride}") from exc
        indices = np.flatnonzero(np.isfinite(scores) & (scores >= conf_threshold))
        for index in indices:
            cx, cy = index % size + 0.5, index // size + 0.5
            left, top, right, bottom = boxes[:, index]
            box = np.array([(cx-left)*stride, (cy-top)*stride,
                            (cx+right)*stride, (cy+bottom)*stride], dtype=np.float32)
            raw = points[:, :, index]
            if not np.isfinite(box).all() or not np.isfinite(raw).all():
                continue
            box[[0, 2]] = np.clip((box[[0, 2]]-pad_x)/scale, 0, orig_w)
            box[[1, 3]] = np.clip((box[[1, 3]]-pad_y)/scale, 0, orig_h)
            coords = np.empty((17, 2), dtype=np.float32)
            coords[:, 0] = np.clip(((raw[:, 0]+cx)*stride-pad_x)/scale, 0, orig_w)
            coords[:, 1] = np.clip(((raw[:, 1]+cy)*stride-pad_y)/scale, 0, orig_h)
            confidence = 1.0/(1.0+np.exp(-np.clip(raw[:, 2], -88, 88)))
            if box[2] <= box[0] or box[3]-box[1] < orig_h*0.10:
                continue
            if np.count_nonzero(confidence > 0.4) < 3:
                continue
            result.append([*map(float, box), float(scores[index]), coords, confidence])
    # Official n-pose weights expose the one-to-many branch. Apply CPU NMS;
    # the end-to-end one-to-one branch can skip this step, but using the same
    # decoder for both keeps the RKNN path deterministic.
    result.sort(key=lambda detection: detection[4], reverse=True)
    keep = []
    for detection in result:
        x1, y1, x2, y2 = detection[:4]
        area = max(0.0, x2-x1) * max(0.0, y2-y1)
        suppressed = False
        for selected in keep:
            sx1, sy1, sx2, sy2 = selected[:4]
            inter = max(0.0, min(x2, sx2)-max(x1, sx1)) * max(0.0, min(y2, sy2)-max(y1, sy1))
            union = area + max(0.0, sx2-sx1)*max(0.0, sy2-sy1) - inter
            if union > 0 and inter/union > 0.45:
                suppressed = True
                break
        if not suppressed:
            keep.append(detection)
    return keep[:100]
