"""Evaluate real YOLO26 pose clips with ONNX; predictions are candidates only."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import resource
import time

import numpy as np
import onnxruntime as ort

from motionbert_input import prepare_input, predictions


def make_windows(snapshots, seconds=8, min_samples=8):
    """Keep track IDs separate; discard absent, stale, restarted or gapped tracks."""
    buffers = defaultdict(deque)
    last_stamp = None
    monitor = None
    for snapshot in snapshots:
        stamp = float(snapshot["timestamp"])
        if snapshot["monitor_pid"] != monitor:
            buffers.clear()
            monitor = snapshot["monitor_pid"]
            last_stamp = None
        if last_stamp is not None and stamp <= last_stamp:
            continue
        last_stamp = stamp
        present = set()
        for person in snapshot["people"]:
            key = (person["track_id"], person["profile_id"])
            present.add(key)
            conf = np.asarray(person["confidence"])
            if conf.shape != (17,) or np.count_nonzero(conf >= 0.3) < 6:
                buffers.pop(key, None)
                continue
            buf = buffers[key]
            if buf and (stamp - buf[-1][0] > 2.5 or snapshot["frame_size"] != buf[-1][2]):
                buf.clear()
            buf.append((stamp, person, snapshot["frame_size"]))
            while buf and stamp - buf[0][0] > seconds:
                buf.popleft()
            if len(buf) >= min_samples and stamp - buf[0][0] >= seconds * 0.75:
                yield key, list(buf)
        for key in list(buffers):
            if key not in present:
                buffers.pop(key)


def evaluate(session, frames, key, clip, names, runs=1):
    raw_points = [p[1]["keypoints"] for p in clip]
    raw_conf = [p[1]["confidence"] for p in clip]
    x = prepare_input(raw_points, raw_conf, clip[-1][2], frames)
    session.run(None, {"skeletons": x})
    cpu_started = time.process_time()
    started = time.perf_counter()
    for _ in range(runs):
        logits = session.run(None, {"skeletons": x})[0]
    elapsed = time.perf_counter() - started
    span = clip[-1][0] - clip[0][0]
    return {
        "track_id": key[0], "profile_id": key[1], "captured_at": clip[-1][0],
        "unique_poses": len(clip), "span_seconds": span,
        "effective_pose_fps": (len(clip) - 1) / span if span else 0,
        "input_frames": frames, "inference_ms": elapsed * 1000 / runs,
        "cpu_ms": (time.process_time() - cpu_started) * 1000 / runs,
        "candidate_actions": predictions(logits, names),
        "experimental": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("clip", type=Path, help="JSONL snapshots from collect_skeletons.py")
    parser.add_argument("--labels", type=Path, default=Path(__file__).with_name("ntu60_labels.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--window-seconds", type=float, default=8)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=0, help="0 evaluates all eligible clips")
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1 or args.min_samples < 2 or args.window_seconds <= 0:
        parser.error("Invalid run/window parameters")
    names = json.loads(args.labels.read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    # Disable idle thread spinning so this experiment does not consume CPU between runs.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - started) * 1000
    shape = session.get_inputs()[0].shape
    if shape[:2] != [1, 2] or shape[3:] != [17, 3]:
        raise ValueError(f"Unexpected input shape: {shape}")
    snapshots = [json.loads(line) for line in args.clip.read_text().splitlines() if line.strip()]
    reports, last_at = [], {}
    for key, clip in make_windows(snapshots, args.window_seconds, args.min_samples):
        if clip[-1][0] - last_at.get(key, 0) < args.window_seconds:
            continue
        report = evaluate(session, shape[2], key, clip, names, args.runs)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        reports.append(report)
        last_at[key] = clip[-1][0]
        if args.max_clips and len(reports) >= args.max_clips:
            break
    # Linux ru_maxrss is KiB; macOS is bytes.
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
    result = {"model": args.model.name, "model_bytes": args.model.stat().st_size,
              "load_ms": load_ms, "threads": args.threads, "runs": args.runs,
              "peak_rss_mb": peak_mb, "snapshots": len(snapshots), "clips": reports,
              "note": "NTU60 closed-set candidates; scores do not prove an action occurred."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}), flush=True)
    if not reports:
        raise SystemExit("No sufficiently continuous observed track; collect another clip")


if __name__ == "__main__":
    main()
