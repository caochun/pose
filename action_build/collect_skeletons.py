"""Collect a bounded keypoint-only experiment from pose's RAM snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("/run/pose/skeleton.json"))
    parser.add_argument("--seconds", type=float, default=45)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    previous = None
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        while time.monotonic() < deadline:
            try:
                snapshot = json.loads(args.source.read_text())
                stamp = snapshot["monitor_pid"], snapshot["timestamp"]
                if stamp != previous and 0 <= time.time() - stamp[1] < 3:
                    output.write(json.dumps(snapshot) + "\n")
                    output.flush()
                    previous = stamp
                    count += 1
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.05)
    print(json.dumps({"snapshots": count, "bytes": args.output.stat().st_size,
                      "output": str(args.output)}), flush=True)
    if not count:
        raise SystemExit("No fresh skeletons received")


if __name__ == "__main__":
    main()
