"""Local multi-person tracking, appearance profiles, and Weixin events."""
from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ReIDExtractor:
    """Optional RKNN OSNet extractor used in shadow mode during calibration."""

    def __init__(self):
        self.mode = os.getenv("POSE_REID_MODE", "off").strip().lower()
        self.model_path = Path(os.getenv(
            "POSE_REID_MODEL",
            "/root/Develop/pose/models/reid/osnet_x0_25-rk3576-i8.rknn",
        ))
        self.rknn = None
        if self.mode not in {"shadow", "on", "enabled"}:
            return
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite(verbose=False)
            if self.rknn.load_rknn(str(self.model_path)) != 0:
                raise RuntimeError("load_rknn failed")
            if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError("init_runtime failed")
            print(f"[ReID] {self.mode} mode enabled: {self.model_path}", flush=True)
        except Exception as exc:
            print(f"[ReID] disabled: {type(exc).__name__}: {exc}", flush=True)
            self.rknn = None

    @property
    def enabled(self):
        return self.rknn is not None

    def embedding(self, frame, box):
        if self.rknn is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
        if x2 - x1 < 20 or y2 - y1 < 30:
            return None
        # The RKNN converter embeds ImageNet mean/std and expects BGR uint8.
        crop = cv2.resize(frame[y1:y2, x1:x2], (128, 256), interpolation=cv2.INTER_LINEAR)
        try:
            output = np.asarray(self.rknn.inference(inputs=[crop[None, ...]])[0], dtype=np.float32)
            vector = output.reshape(-1)
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm > 1e-6 else None
        except Exception as exc:
            print(f"[ReID] inference failed: {type(exc).__name__}", flush=True)
            return None

    def close(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


class WeixinNotifier:
    def __init__(self, target="weixin", hermes_bin=None):
        self.target = target
        self.hermes_bin = hermes_bin or shutil.which("hermes") or "/root/.hermes/venvs/hermes/bin/hermes"
        self._q = queue.Queue(maxsize=16)
        self._failed_media = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="weixin-notify", daemon=True)
        self._thread.start()

    def send(self, text, media=None, context=None):
        try:
            # Keep the attachment path separate until Hermes has uploaded it.
            self._q.put_nowait((text, media, context or {}))
        except queue.Full:
            print("[Weixin] notification queue full; dropping event", flush=True)
            if media:
                Path(media).unlink(missing_ok=True)

    def _worker(self):
        while not self._stop.is_set():
            try:
                text, media, context = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            helper = "/root/Develop/pose/hermes_plugins/pose_queries/vision.py"
            python = os.getenv("HERMES_PYTHON", "/root/.hermes/venvs/hermes/bin/python")
            delivered = False
            # iLink sessions can recover when the user sends a new inbound message.
            # Keep the event attachment and retry across that session refresh.
            for attempt in range(1, 6):
                try:
                    p = subprocess.run(
                        [python, helper, "event", "--target", self.target,
                         "--text", text, "--media", str(Path(media).resolve()) if media else "",
                         "--context", json.dumps(context, ensure_ascii=False)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, timeout=75, check=False,
                    )
                    if p.returncode == 0:
                        delivered = True
                        print("[Weixin] event sent", flush=True)
                        if media:
                            try:
                                Path(media).unlink(missing_ok=True)
                                print(f"[Weixin] attachment deleted: {media}", flush=True)
                            except Exception as exc:
                                print(f"[Weixin] attachment cleanup failed: {exc}", flush=True)
                        break
                    print(f"[Weixin] send failed attempt={attempt}/5 rc={p.returncode}: {p.stdout[-500:]}", flush=True)
                except Exception as exc:
                    print(f"[Weixin] send exception attempt={attempt}/5: {type(exc).__name__}", flush=True)
                if attempt < 5:
                    print("[Weixin] retrying event after 60s; send a message to refresh the Weixin session", flush=True)
                    if self._stop.wait(60):
                        break
            if not delivered:
                self._retain_failure(media)

    def _retain_failure(self, media):
        if not media:
            return
        self._failed_media.append(Path(media))
        # Bound retained failed attachments; the user can request a fresh screenshot.
        while len(self._failed_media) > 8:
            self._failed_media.pop(0).unlink(missing_ok=True)

    def close(self):
        self._stop.set()


class ProfileStore:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("""CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT, feature TEXT NOT NULL,
            first_seen REAL NOT NULL, last_seen REAL NOT NULL,
            seen_count INTEGER NOT NULL DEFAULT 1,
            last_pose TEXT NOT NULL DEFAULT 'unknown')""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER,
            event TEXT NOT NULL, pose TEXT, created_at REAL NOT NULL,
            detail TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS reid_features (
            profile_id INTEGER PRIMARY KEY, embedding TEXT NOT NULL,
            updated_at REAL NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS reid_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,
            track_id INTEGER NOT NULL, assigned_profile_id INTEGER,
            nearest_profile_id INTEGER, similarity REAL NOT NULL,
            mode TEXT NOT NULL)""")
        # A single cross-process collection session is controlled by the
        # Hermes command plugin and consumed by the monitor process.
        self.db.execute("""CREATE TABLE IF NOT EXISTS reid_collection (
            id INTEGER PRIMARY KEY CHECK(id=1), active INTEGER NOT NULL DEFAULT 0,
            profile_id INTEGER, track_id INTEGER, started_at REAL,
            updated_at REAL NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS reid_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id INTEGER NOT NULL,
            track_id INTEGER, created_at REAL NOT NULL, pose TEXT NOT NULL,
            box TEXT NOT NULL, embedding TEXT NOT NULL)""")
        self.db.commit()

    @staticmethod
    def _cosine(a, b):
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 1e-6 else 0.0

    def match_or_create(self, feature, now, threshold=0.88, exclude_ids=()):
        with self._lock:
            rows = self.db.execute("SELECT id, feature FROM profiles").fetchall()
            best_id, best_sim = None, -1.0
            excluded = set(exclude_ids)
            for pid, raw in rows:
                if int(pid) in excluded:
                    continue
                try:
                    sim = self._cosine(feature, json.loads(raw))
                except Exception:
                    continue
                if sim > best_sim:
                    best_id, best_sim = int(pid), sim
            if best_id is not None and best_sim >= threshold:
                return best_id, best_sim, False
            cur = self.db.execute(
                "INSERT INTO profiles(feature, first_seen, last_seen) VALUES(?,?,?)",
                (json.dumps(list(map(float, feature))), now, now))
            self.db.commit()
            return int(cur.lastrowid), 1.0, True

    def update(self, profile_id, feature, now, pose):
        with self._lock:
            row = self.db.execute("SELECT feature, seen_count FROM profiles WHERE id=?", (profile_id,)).fetchone()
            if row:
                old = np.asarray(json.loads(row[0]), dtype=np.float32)
                merged = (old * 0.85 + np.asarray(feature, dtype=np.float32) * 0.15).tolist()
                self.db.execute("UPDATE profiles SET feature=?, last_seen=?, seen_count=?, last_pose=? WHERE id=?",
                                 (json.dumps(merged), now, int(row[1]) + 1, pose, profile_id))
                self.db.commit()

    def event(self, profile_id, kind, pose, now, detail=""):
        with self._lock:
            self.db.execute("INSERT INTO events(profile_id,event,pose,created_at,detail) VALUES(?,?,?,?,?)",
                            (profile_id, kind, pose, now, detail))
            self.db.commit()

    def nearest_reid(self, embedding):
        with self._lock:
            rows = self.db.execute("SELECT profile_id, embedding FROM reid_features").fetchall()
        best_id, best_sim = None, -1.0
        for pid, raw in rows:
            try:
                sim = self._cosine(embedding, json.loads(raw))
            except Exception:
                continue
            if sim > best_sim:
                best_id, best_sim = int(pid), sim
        return best_id, best_sim

    def save_reid(self, profile_id, embedding, now):
        values = list(map(float, np.asarray(embedding, dtype=np.float32)))
        with self._lock:
            row = self.db.execute(
                "SELECT embedding FROM reid_features WHERE profile_id=?", (profile_id,)
            ).fetchone()
            if row:
                old = np.asarray(json.loads(row[0]), dtype=np.float32)
                merged = old * 0.85 + np.asarray(values, dtype=np.float32) * 0.15
                norm = float(np.linalg.norm(merged))
                values = (merged / norm if norm > 1e-6 else merged).tolist()
                self.db.execute(
                    "UPDATE reid_features SET embedding=?, updated_at=? WHERE profile_id=?",
                    (json.dumps(values), now, profile_id),
                )
            else:
                self.db.execute(
                    "INSERT INTO reid_features(profile_id, embedding, updated_at) VALUES(?,?,?)",
                    (profile_id, json.dumps(values), now),
                )
            self.db.commit()

    def reid_observation(self, now, track_id, assigned_id, nearest_id, similarity, mode="shadow"):
        with self._lock:
            self.db.execute(
                "INSERT INTO reid_observations(created_at,track_id,assigned_profile_id,nearest_profile_id,similarity,mode) VALUES(?,?,?,?,?,?)",
                (now, track_id, assigned_id, nearest_id, float(similarity), mode),
            )
            self.db.commit()

    def collection_state(self):
        with self._lock:
            row = self.db.execute(
                "SELECT active,profile_id,track_id,started_at FROM reid_collection WHERE id=1"
            ).fetchone()
        if not row or not row[0] or row[1] is None:
            return None
        return {"profile_id": int(row[1]), "track_id": int(row[2]) if row[2] is not None else None,
                "started_at": float(row[3] or 0)}

    def save_reid_sample(self, profile_id, track_id, embedding, now, pose, box):
        values = list(map(float, np.asarray(embedding, dtype=np.float32)))
        with self._lock:
            self.db.execute(
                "INSERT INTO reid_samples(profile_id,track_id,created_at,pose,box,embedding) VALUES(?,?,?,?,?,?)",
                (int(profile_id), int(track_id), float(now), pose or "unknown",
                 json.dumps(list(map(float, box))), json.dumps(values)),
            )
            self.db.execute("UPDATE reid_collection SET updated_at=? WHERE id=1", (float(now),))
            self.db.commit()

    def close(self):
        with self._lock:
            self.db.close()


def appearance_feature(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(34, dtype=np.float32)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten().astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm > 1e-6:
        hist /= norm
    return np.concatenate([hist, np.asarray([(x2-x1)/max(w,1), (y2-y1)/max(h,1)], dtype=np.float32)])


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(aa + bb - inter, 1e-6)


def _center_distance(a, b, frame_shape):
    h, w = frame_shape[:2]
    ac = ((a[0]+a[2])/2, (a[1]+a[3])/2)
    bc = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    return float(np.hypot(ac[0]-bc[0], ac[1]-bc[1]) / max(np.hypot(w,h), 1.0))


def _pose_from_detection(det):
    kp, confs, box = np.asarray(det[5]), np.asarray(det[6]), det[:4]
    hips = [kp[i][1] for i in (11,12) if confs[i] > 0.3]
    knees = [kp[i][1] for i in (13,14) if confs[i] > 0.3]
    bh = float(box[3]-box[1])
    if not hips or not knees or bh < 1:
        return "unknown"
    return "standing" if (float(np.mean(knees))-float(np.mean(hips)))/bh > 0.15 else "sitting"


class PersonEventManager:
    # A missing pose detection does not prove physical departure, especially
    # for a seated/occluded person at 1 inference per second.
    EXIT_GAP = 30.0
    MIN_MISSED_RESULTS = 20
    POSTURE_COOLDOWN = 300.0

    def __init__(self, db_path, media_dir, notifier=None):
        self.store = ProfileStore(db_path)
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        # Clean owned leftovers from earlier notifier runs, retaining the latest eight.
        old_media = sorted(self.media_dir.glob("*_multi_person.jpg"), key=lambda p: p.stat().st_mtime)
        for old in old_media[:-8]:
            old.unlink(missing_ok=True)
        self.notifier = notifier or WeixinNotifier()
        if isinstance(self.notifier, WeixinNotifier):
            self.notifier._failed_media = list(old_media[-8:])
        self.tracks = {}
        self.reid = ReIDExtractor()
        self.next_track_id = 1
        self.labels = []
        self._last_diag = 0.0
        self._inference_at = 0.0
        self._frame_size = (0, 0)
        self._last_status = 0.0
        self._collection_last_at = 0.0
        self._collection_last_box = None
        self._collection_last_pose = None
        self._collection_warn_at = 0.0

    def _new_track(self, det, feature, pose, now, active_profiles):
        pid, similarity, is_new = self.store.match_or_create(feature, now, exclude_ids=active_profiles)
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = {"profile_id": pid, "box": tuple(map(float, det[:4])),
                            "feature": feature, "pose": pose, "last_seen": now,
                            "last_posture_event": now, "similarity": similarity,
                            "missed_results": 0}
        kind = "new_profile" if is_new else "person_appeared"
        self.store.event(pid, kind, pose, now, f"track={tid};similarity={similarity:.3f}")
        return tid, kind, pid, pose, similarity, det[:4]

    def _collect_sample(self, frame, tid, pid, pose, box, now, collection, person_count):
        if not (collection and person_count == 1 and pid == collection["profile_id"] and self.reid.enabled):
            return
        embedding = self.reid.embedding(frame, box)
        if embedding is None:
            return
        box = tuple(map(float, box))
        changed = self._collection_last_box is None
        if self._collection_last_box is not None:
            old = self._collection_last_box
            old_area = max((old[2]-old[0])*(old[3]-old[1]), 1.0)
            new_area = max((box[2]-box[0])*(box[3]-box[1]), 1.0)
            area_ratio = max(new_area, old_area) / min(new_area, old_area)
            center_shift = _center_distance(old, box, frame.shape)
            changed = area_ratio >= 1.25 or center_shift >= 0.08 or pose != self._collection_last_pose
        if (now - self._collection_last_at >= 8.0) or changed:
            self.store.save_reid_sample(collection["profile_id"], tid, embedding, now, pose, box)
            self._collection_last_at = now
            self._collection_last_box = box
            self._collection_last_pose = pose
            print(f"[ReID collection] saved P{collection['profile_id']} sample pose={pose} changed={changed}", flush=True)

    def process(self, frame, detections, _global_pose=None, now=None):
        now = time.time() if now is None else now
        self._inference_at = now
        self._frame_size = (frame.shape[1], frame.shape[0])
        detections = list(detections or [])
        prepared = [(d, appearance_feature(frame, d[:4]), _pose_from_detection(d)) for d in detections]
        collection = self.store.collection_state()
        # Collection is deliberately conservative: the command promises a
        # one-person scene, so any multi-person inference is discarded.
        if collection and len(prepared) != 1:
            if now - self._collection_warn_at >= 30:
                print(f"[ReID collection] skipped frame: expected one person, got {len(prepared)}", flush=True)
                self._collection_warn_at = now
        old_ids = list(self.tracks)
        candidates = []
        for tid in old_ids:
            tr = self.tracks[tid]
            for idx, (det, feat, _pose) in enumerate(prepared):
                sim = ProfileStore._cosine(tr["feature"], feat)
                iou = _iou(tr["box"], det[:4])
                dist = _center_distance(tr["box"], det[:4], frame.shape)
                if (iou >= 0.08 or (sim >= 0.82 and dist <= 0.28)) and dist <= 0.45:
                    candidates.append((0.60*iou + 0.40*sim, tid, idx))
        candidates.sort(reverse=True)
        matched_tracks, matched_dets, assignments = set(), set(), {}
        for _score, tid, idx in candidates:
            if tid not in matched_tracks and idx not in matched_dets:
                matched_tracks.add(tid); matched_dets.add(idx); assignments[idx] = tid

        events = []
        # Reserve profiles even while their tracks are temporarily missing.
        # Otherwise two live tracks can claim P3 and one produces a false
        # departure while the other is still visible.
        active_profiles = {tr["profile_id"] for tr in self.tracks.values()}
        for idx, (det, feat, pose) in enumerate(prepared):
            if idx not in assignments:
                tid, kind, pid, p, sim, box = self._new_track(det, feat, pose, now, active_profiles)
                active_profiles.add(pid); assignments[idx] = tid
                if self.reid.enabled:
                    embedding = self.reid.embedding(frame, det[:4])
                    if embedding is not None:
                        nearest_id, reid_sim = self.store.nearest_reid(embedding)
                        self.store.reid_observation(now, tid, pid, nearest_id, reid_sim)
                        nearest = f"P{nearest_id}" if nearest_id is not None else "none"
                        print(f"[ReID shadow] T{tid} assigned P{pid}; nearest={nearest} sim={reid_sim:.3f}", flush=True)
                        self.store.save_reid(pid, embedding, now)
                self._collect_sample(frame, tid, pid, p, box, now, collection, len(prepared))
                events.append((kind, tid, pid, p, sim, box))
                continue
            tid = assignments[idx]
            tr = self.tracks[tid]
            gap = now - tr["last_seen"]
            if tr["missed_results"] and gap >= 5:
                print(f'[Tracker] recovered P{tr["profile_id"]}/T{tid}: '
                      f'gap={gap:.1f}s missed={tr["missed_results"]}', flush=True)
            tr["missed_results"] = 0
            old_pose = tr["pose"]
            tr["box"] = tuple(map(float, det[:4]))
            tr["feature"] = tr["feature"] * 0.85 + feat * 0.15
            tr["last_seen"], tr["pose"] = now, pose
            if pose in ("sitting","standing") and pose != old_pose and now-tr["last_posture_event"] >= self.POSTURE_COOLDOWN:
                tr["last_posture_event"] = now
                self.store.event(tr["profile_id"], "posture_changed", pose, now, f"track={tid}")
                events.append(("posture_changed", tid, tr["profile_id"], pose, tr["similarity"], det[:4]))
            self.store.update(tr["profile_id"], tr["feature"], now, pose)

            self._collect_sample(frame, tid, tr["profile_id"], pose, det[:4], now,
                                 collection, len(prepared))

        for tid in old_ids:
            if tid in matched_tracks:
                continue
            tr = self.tracks[tid]
            tr["missed_results"] += 1
            gap = now - tr["last_seen"]
            if gap >= self.EXIT_GAP and tr["missed_results"] >= self.MIN_MISSED_RESULTS:
                self.tracks.pop(tid)
                detail = f'track={tid};gap_seconds={gap:.1f};missed_results={tr["missed_results"]}'
                self.store.event(tr["profile_id"], "person_unobserved", tr["pose"], now, detail)
                print(f'[Tracker] unobserved P{tr["profile_id"]}: {detail}', flush=True)
                events.append(("person_unobserved", tid, tr["profile_id"], tr["pose"], tr["similarity"], None))

        if now - self._last_diag >= 30:
            missing = sum(tr["missed_results"] > 0 for tr in self.tracks.values())
            print(f'[Tracker] detections={len(prepared)} tracks={len(self.tracks)} '
                  f'missing={missing}', flush=True)
            self._last_diag = now

        self.labels = []
        for idx, (_det, _feat, pose) in enumerate(prepared):
            tid = assignments[idx]; tr = self.tracks[tid]
            self.labels.append((tid, tr["profile_id"], pose))
        self._publish_skeletons(detections, now)
        return events

    def _publish_skeletons(self, detections, now):
        """Latest observed poses in RAM for independent temporal-model experiments."""
        path = Path("/run/pose/skeleton.json")
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = {
                "timestamp": now, "monitor_pid": os.getpid(),
                "frame_size": self._frame_size,
                "people": [
                    {"track_id": tid, "profile_id": pid, "pose": pose,
                     "box": list(map(float, det[:4])),
                     "keypoints": np.asarray(det[5]).tolist(),
                     "confidence": np.asarray(det[6]).tolist()}
                    for det, (tid, pid, pose) in zip(detections, self.labels)
                ],
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[Skeleton] publish failed: {type(exc).__name__}", flush=True)

    def publish_status(self, stream_fps, inference_fps, now=None, model_name=None):
        now = time.time() if now is None else now
        if now - self._last_status < 1.0:
            return
        self._last_status = now
        path = Path("/run/pose/status.json")
        tmp = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = {"timestamp": now, "inference_at": self._inference_at,
                       "pose_model": model_name,
                       "stream_fps": float(stream_fps), "inference_fps": float(inference_fps),
                       "frame_size": self._frame_size,
                       "tracks": [{"track_id": tid, "profile_id": tr["profile_id"],
                                   "pose": tr["pose"], "last_seen": tr["last_seen"], "box": tr["box"],
                                   "missed_results": tr["missed_results"]}
                                  for tid, tr in self.tracks.items()]}
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[Status] publish failed: {type(exc).__name__}", flush=True)

    def emit(self, events, frame):
        if not events:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.media_dir / f"{stamp}_multi_person.jpg"
        media = str(path) if cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY),82]) else None
        lines = []
        for kind, tid, pid, pose, similarity, _box in events:
            if kind == "new_profile":
                lines.append(f"P{pid}（轨迹T{tid}）新建人物档案，姿态：{pose}")
            elif kind == "person_appeared":
                lines.append(f"P{pid}（轨迹T{tid}）出现，姿态：{pose}，匹配度：{similarity:.2f}")
            elif kind == "person_unobserved":
                lines.append(f"P{pid}（轨迹T{tid}）持续至少30秒未检测到，可能被遮挡或已离开；最后姿态：{pose}")
            else:
                lines.append(f"P{pid}（轨迹T{tid}）姿态变化：{pose}")
        context = {"captured_at": time.time(), "profile_ids": sorted({ev[2] for ev in events}),
                   "events": [{"kind": ev[0], "track_id": ev[1], "profile_id": ev[2]} for ev in events],
                   "detected": [{"track_id": tid, "profile_id": pid, "pose": pose}
                                for tid, pid, pose in self.labels]}
        self.notifier.send(f"人物事件（{len(events)}）\n" + "\n".join(lines), media, context)

    def close(self):
        self.reid.close(); self.notifier.close(); self.store.close()
