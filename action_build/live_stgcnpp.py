"""Run the pretrained MMAction2 ST-GCN++ NTU60 model on pose snapshots.

This is an A/B worker for the existing MotionBERT service.  It keeps the same
status schema so the two workers can be compared without changing the pose
monitor or Weixin integration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import signal
import time

import numpy as np
import onnxruntime as ort

from live_motionbert import LiveFeed, follow
from run_motionbert import make_windows


def prepare_input(clip, frames=100):
    """Apply MMAction2 PreNormalize2D and FormatGCNInput for one person."""
    xy = np.asarray([row[1]['keypoints'] for row in clip], dtype=np.float32)
    conf = np.asarray([row[1]['confidence'] for row in clip], dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1:] != (17, 2) or conf.shape != xy.shape[:2]:
        raise ValueError('Expected a continuous T x 17 x 2/17 keypoint clip')
    width, height = clip[-1][2]
    if width <= 0 or height <= 0:
        raise ValueError('Invalid frame size')
    xy = xy.copy()
    xy[..., 0] = (xy[..., 0] - width / 2) / (width / 2)
    xy[..., 1] = (xy[..., 1] - height / 2) / (height / 2)
    motion = np.concatenate((xy, np.clip(conf[..., None], 0, 1)), axis=-1)
    # UniformSampleFrames(test_mode=True) has the same useful property here:
    # long clips are downsampled and short clips repeat nearby observations.
    indices = np.linspace(0, len(motion), num=frames, endpoint=False, dtype=int)
    motion = motion[indices]
    motion = np.stack((motion, np.zeros_like(motion)), axis=0)
    return motion[None].astype(np.float32, copy=False)


def evaluate(session, clip, names, frames=100):
    x = prepare_input(clip, frames)
    started = time.perf_counter()
    logits = session.run(None, {'skeletons': x})[0]
    elapsed = (time.perf_counter() - started) * 1000
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if len(logits) != len(names) or not np.isfinite(logits).all():
        raise ValueError('Invalid ST-GCN++ output')
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    candidates = [
        {'id': int(i), 'label': names[i], 'score': float(probs[i])}
        for i in np.argsort(probs)[-5:][::-1]
    ]
    span = clip[-1][0] - clip[0][0]
    return {
        'track_id': clip[-1][1]['track_id'],
        'profile_id': clip[-1][1]['profile_id'],
        'captured_at': clip[-1][0],
        'unique_poses': len(clip),
        'span_seconds': span,
        'effective_pose_fps': (len(clip) - 1) / span if span else 0,
        'input_frames': frames,
        'inference_ms': elapsed,
        'cpu_ms': elapsed,
        'candidate_actions': candidates,
        'experimental': True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--source', type=Path, default=Path('/run/pose/skeleton.json'))
    parser.add_argument('--output', type=Path, default=Path('/run/pose/stgcn_actions.json'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    if args.threads < 1 or args.interval < 5:
        parser.error('threads must be positive and interval at least 5 seconds')
    feed, stop = LiveFeed(interval=args.interval), __import__('threading').Event()
    feed.model_name = 'ST-GCN++ NTU60 2D'
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    import threading
    reader = threading.Thread(target=follow, args=(feed, args.source, args.output, stop), daemon=True)
    reader.start()
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        session = ort.InferenceSession(str(args.model), options, providers=['CPUExecutionProvider'])
        shape = session.get_inputs()[0].shape
        if shape != [1, 2, 100, 17, 3]:
            raise ValueError(f'Unexpected model input: {shape}')
        names = json.loads(Path(__file__).with_name('ntu60_labels.json').read_text())
        with feed.lock:
            feed.loaded = True
        print(f'[Action] loaded {args.model.name}; ST-GCN++ CPU threads={args.threads}; interval={args.interval}s', flush=True)
        while not stop.is_set():
            if not reader.is_alive():
                raise RuntimeError('Skeleton reader stopped')
            now, monotonic = time.time(), time.monotonic()
            with feed.lock:
                ready = None
                if feed.loaded and not feed.busy and monotonic >= feed.next_run and 0 <= now - feed.source_at < 3:
                    if feed.ready:
                        key = min(feed.ready, key=lambda k: feed.last_started.get(k, 0))
                        ready = (key, feed.versions[key], feed.ready[key])
                        feed.busy = key
                        feed.last_started[key] = monotonic
                        feed.next_run = monotonic + feed.interval
            if ready is None:
                stop.wait(0.1)
                continue
            key, version, clip = ready
            try:
                result = evaluate(session, clip, names)
                accepted = feed.complete((key, version, clip), result, time.time())
                top = result['candidate_actions'][0]
                print(f'[Action] P{key[2]}/T{key[1]} candidate={top["label"]} score={top["score"]:.3f} '
                      f'infer_ms={result["inference_ms"]:.0f} accepted={accepted}', flush=True)
            except Exception as exc:
                with feed.lock:
                    feed.busy = None
                    feed.last_error = str(exc)[:200]
                print(f'[Action] inference failed: {type(exc).__name__}: {exc}', flush=True)
    finally:
        stop.set()
        reader.join(timeout=2)
        args.output.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
