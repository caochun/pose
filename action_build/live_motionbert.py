"""Independent, bounded MotionBERT worker consuming pose snapshots in RAM."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import threading
import time

import onnxruntime as ort

from run_motionbert import evaluate, make_windows


class LiveFeed:
    def __init__(self, window_seconds=8, min_samples=8, interval=15, max_age=30):
        self.window_seconds, self.min_samples = window_seconds, min_samples
        self.interval, self.max_age = interval, max_age
        self.lock = threading.Lock()
        self.history = deque(maxlen=256)
        self.current = {}
        self.versions = {}
        self.ready = {}
        self.results = {}
        self.last_started = {}
        self.sequence = 0
        self.stamp = None
        self.source_at = 0
        self.next_run = 0
        self.busy = None
        self.loaded = False
        self.last_error = None

    def accept(self, snapshot):
        stamp = snapshot['monitor_pid'], snapshot['timestamp']
        with self.lock:
            if stamp == self.stamp:
                return
            if self.stamp and (stamp[0] != self.stamp[0] or stamp[1] <= self.stamp[1]
                               or stamp[1] - self.stamp[1] > 2.5):
                self.current.clear()
                self.history.clear()
            self.stamp, self.source_at = stamp, stamp[1]
            self.history.append(snapshot)
            while self.history and stamp[1] - self.history[0]['timestamp'] > self.window_seconds * 2:
                self.history.popleft()
            current = {(stamp[0], p['track_id'], p['profile_id']): p for p in snapshot['people']}
            versions = {}
            for key in current:
                if key in self.current:
                    versions[key] = self.versions[key]
                else:
                    self.sequence += 1
                    versions[key] = self.sequence
            self.current, self.versions = current, versions
            self.ready = {}
            for key, clip in make_windows(self.history, self.window_seconds, self.min_samples):
                full_key = (stamp[0], *key)
                if full_key in current and clip[-1][0] == stamp[1]:
                    self.ready[full_key] = clip
            self.results = {key: value for key, value in self.results.items()
                            if key in current and value[0] == versions[key]}
            self.last_started = {key: value for key, value in self.last_started.items() if key in current}

    def stale(self):
        with self.lock:
            self.current.clear()
            self.versions.clear()
            self.ready.clear()
            self.results.clear()
            self.last_started.clear()
            self.history.clear()

    def job(self, now, monotonic):
        with self.lock:
            if not self.loaded or self.busy or monotonic < self.next_run or not 0 <= now - self.source_at < 3:
                return None
            if not self.ready:
                return None
            key = min(self.ready, key=lambda k: self.last_started.get(k, 0))
            clip = self.ready[key]
            job = (key, self.versions[key], clip)
            self.busy = key
            self.last_started[key] = monotonic
            self.next_run = monotonic + self.interval
            return job

    def complete(self, job, result, now):
        key, version, clip = job
        with self.lock:
            self.busy = None
            if self.versions.get(key) != version or not 0 <= now - clip[-1][0] <= self.max_age:
                return False
            result['completed_at'] = now
            self.results[key] = version, result
            self.last_error = None
            return True

    def status(self, now):
        with self.lock:
            fresh = 0 <= now - self.source_at < 3
            rows = []
            if fresh:
                for key, person in self.current.items():
                    saved = self.results.get(key)
                    result = saved[1] if saved and 0 <= now - saved[1]['captured_at'] <= self.max_age else None
                    rows.append({'track_id': key[1], 'profile_id': key[2], 'pose': person['pose'],
                                 'state': 'candidate' if result else 'collecting',
                                 'result': result})
            return {'timestamp': now, 'source_at': self.source_at, 'source_fresh': fresh,
                    'model': getattr(self, 'model_name', 'MotionBERT NTU60'),
                    'backend': 'CPU', 'loaded': self.loaded,
                    'interval_seconds': self.interval, 'window_seconds': self.window_seconds,
                    'max_result_age': self.max_age, 'busy': self.busy is not None,
                    'last_error': self.last_error, 'people': rows, 'experimental': True}


def follow(feed, source, output, stop):
    last_write = 0
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    while not stop.is_set():
        now = time.time()
        try:
            snapshot = json.loads(source.read_text())
            if 0 <= now - snapshot['timestamp'] < 3:
                feed.accept(snapshot)
            else:
                feed.stale()
        except (OSError, ValueError, KeyError, TypeError):
            feed.stale()
        if time.monotonic() - last_write >= 1:
            tmp = output.with_suffix('.tmp')
            tmp.write_text(json.dumps(feed.status(now)), encoding='utf-8')
            os.replace(tmp, output)
            last_write = time.monotonic()
        stop.wait(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--source', type=Path, default=Path('/run/pose/skeleton.json'))
    parser.add_argument('--output', type=Path, default=Path('/run/pose/actions.json'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    if args.threads < 1 or args.interval < 5:
        parser.error('threads must be positive and interval at least 5 seconds')
    feed, stop = LiveFeed(interval=args.interval), threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
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
        if shape != [1, 2, 32, 17, 3]:
            raise ValueError(f'Unexpected model input: {shape}')
        names = json.loads(Path(__file__).with_name('ntu60_labels.json').read_text())
        with feed.lock:
            feed.loaded = True
        print(f'[Action] loaded {args.model.name}; CPU threads={args.threads}; interval={args.interval}s', flush=True)
        while not stop.is_set():
            if not reader.is_alive():
                raise RuntimeError('Skeleton reader stopped')
            job = feed.job(time.time(), time.monotonic())
            if job is None:
                stop.wait(0.1)
                continue
            try:
                key, _version, clip = job
                result = evaluate(session, 32, key[1:], clip, names, warmup=False)
                accepted = feed.complete(job, result, time.time())
                top = result['candidate_actions'][0]
                print(f'[Action] P{key[2]}/T{key[1]} candidate={top["label"]} '
                      f'score={top["score"]:.3f} infer_ms={result["inference_ms"]:.0f} accepted={accepted}', flush=True)
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
