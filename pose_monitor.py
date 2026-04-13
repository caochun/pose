"""
姿态监控：检测画面中的人是坐着还是站立，并记录时长
使用 GStreamer appsink + YOLOv8-pose RKNN NPU 推理

架构：
- 主进程：GStreamer采集 + 推流（永不重启，推流不中断）
- 推理子进程：RKNN推理隔离，RSS超限时独立重启，主进程无感知

优化：
- 异步推理：推流帧率不受 NPU 推理延迟影响
- NV12 直通：省去 BGR→NV12 软件 videoconvert
- 推理子进程隔离：librknnrt.so 内存只在子进程积累，重启不影响推流
"""
import os
os.environ['RKNN_LOG_LEVEL'] = '0'

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
import time
import csv
import sys
import threading
import argparse
import ctypes
import multiprocessing as mp
from datetime import datetime

# ─── glibc 调优（主进程） ───────────────────────────────────────
try:
    _libc = ctypes.cdll.LoadLibrary('libc.so.6')
    _libc.mallopt(ctypes.c_int(-8), ctypes.c_int(2))  # MALLOC_ARENA_MAX=2
    def malloc_trim(): _libc.malloc_trim(0)
except Exception:
    def malloc_trim(): pass

Gst.init(None)

# ─── YOLOv8-pose 关键点索引 (COCO 17点) ───
KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE = 0, 1, 2
KP_LEFT_EAR, KP_RIGHT_EAR = 3, 4
KP_LEFT_SHLDR, KP_RIGHT_SHLDR = 5, 6
KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12
KP_LEFT_KNEE, KP_RIGHT_KNEE = 13, 14
KP_LEFT_ANKLE, KP_RIGHT_ANKLE = 15, 16

INPUT_SIZE  = 640
NMS_THRESH  = 0.45
CONF_THRESH = 0.35


def letterbox(img, target_size=640):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_y = (target_size - nh) // 2
    pad_x = (target_size - nw) // 2
    padded[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_resized
    return padded, scale, pad_x, pad_y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def decode_outputs(outputs, scale, pad_x, pad_y, orig_w, orig_h):
    keypoints_all = outputs[3]
    boxes = []
    strides, grid_sizes = [8, 16, 32], [80, 40, 20]
    for i, (stride, gs) in enumerate(zip(strides, grid_sizes)):
        feat = outputs[i].reshape(1, 65, -1)
        bbox = feat[0, :64, :]
        conf = sigmoid(feat[0, 64, :])
        idx_start = sum(g*g for g in grid_sizes[:i])
        kps = keypoints_all[0, :, :, idx_start:idx_start+gs*gs]
        for ci in range(gs * gs):
            if conf[ci] < CONF_THRESH:
                continue
            cx, cy = ci % gs, ci // gs
            b = softmax(bbox[:, ci].reshape(4, 16), axis=1)
            b = (b * np.arange(16)).sum(axis=1)
            x1 = max(0, ((cx + 0.5 - b[0]) * stride - pad_x) / scale)
            y1 = max(0, ((cy + 0.5 - b[1]) * stride - pad_y) / scale)
            x2 = min(orig_w, ((cx + 0.5 + b[2]) * stride - pad_x) / scale)
            y2 = min(orig_h, ((cy + 0.5 + b[3]) * stride - pad_y) / scale)
            kp = kps[:, :, ci]
            kp_coords = kp[:, :2].copy()
            kp_coords[:, 0] = (kp_coords[:, 0] - pad_x) / scale
            kp_coords[:, 1] = (kp_coords[:, 1] - pad_y) / scale
            boxes.append([x1, y1, x2, y2, float(conf[ci]), kp_coords, kp[:, 2]])
    if not boxes:
        return []
    scores = np.array([b[4] for b in boxes])
    indices = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
        scores.tolist(), CONF_THRESH, NMS_THRESH)
    return [boxes[i] for i in indices.flatten()] if len(indices) else []


def classify_pose(kp_coords, kp_confs, box):
    MIN_CONF = 0.3
    hip_ys, knee_ys = [], []
    for kp in [KP_LEFT_HIP, KP_RIGHT_HIP]:
        if kp_confs[kp] > MIN_CONF: hip_ys.append(kp_coords[kp][1])
    for kp in [KP_LEFT_KNEE, KP_RIGHT_KNEE]:
        if kp_confs[kp] > MIN_CONF: knee_ys.append(kp_coords[kp][1])
    if not hip_ys or not knee_ys:
        return 'unknown'
    box_h = box[3] - box[1]
    if box_h < 1:
        return 'unknown'
    return 'standing' if (np.mean(knee_ys) - np.mean(hip_ys)) / box_h > 0.15 else 'sitting'


def _fmt_duration(seconds):
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def draw_detections(frame, detections, pose, sitting_total=0.0, standing_total=0.0):
    # 左上角统计信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (label, val, color) in enumerate([
        (f'Sit  {_fmt_duration(sitting_total)}',  sitting_total,  (0, 165, 255)),
        (f'Std  {_fmt_duration(standing_total)}', standing_total, (0, 255,   0)),
    ]):
        y = 32 + i * 36
        cv2.putText(frame, label, (8, y), font, 0.9, (0, 0, 0),   4, cv2.LINE_AA)
        cv2.putText(frame, label, (8, y), font, 0.9, color,        2, cv2.LINE_AA)

    if not detections:
        return
    best = max(detections, key=lambda d: d[4])
    x1, y1, x2, y2 = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    color = (0,255,0) if pose=='standing' else (0,165,255) if pose=='sitting' else (128,128,128)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, pose, (x1, y1-10), font, 0.8, color, 2)
    for ki in range(17):
        if best[6][ki] > 0.3:
            cv2.circle(frame, (int(best[5][ki][0]), int(best[5][ki][1])), 4, (255,0,0), -1)


def _rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


# ═══════════════════════════════════════════════════════════════
# 推理子进程：独立运行 RKNN，RSS 超限时自行退出，由主进程重启
# ═══════════════════════════════════════════════════════════════
def _inference_process(model_path, frame_queue, result_queue,
                       width, height, max_rss_mb):
    """
    推理子进程入口。
    frame_queue:  主进程 → 子进程，item = (nv12_bytes, orig_w, orig_h, sitting_total, standing_total)
    result_queue: 子进程 → 主进程，item = (annotated_nv12_bytes, pose_str)
                  或 None 表示子进程即将退出
    """
    os.environ['RKNN_LOG_LEVEL'] = '0'

    # 子进程也做 malloc 调优
    try:
        import ctypes as _ct
        _lc = _ct.cdll.LoadLibrary('libc.so.6')
        _lc.mallopt(_ct.c_int(-8), _ct.c_int(1))  # 子进程单线程，1个arena
        def _trim(): _lc.malloc_trim(0)
    except Exception:
        def _trim(): pass

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(model_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    print(f'[InferProc:{os.getpid()}] RKNN loaded, max_rss={max_rss_mb}MB', flush=True)

    infer_count = 0
    while True:
        item = frame_queue.get()
        if item is None:
            break

        nv12_bytes, orig_w, orig_h, sitting_total, standing_total = item
        nv12 = np.frombuffer(nv12_bytes, dtype=np.uint8).reshape(orig_h * 3 // 2, orig_w)

        try:
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            inp, scale, pad_x, pad_y = letterbox(bgr, INPUT_SIZE)
            inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            outputs = rknn.inference(inputs=[inp_rgb[np.newaxis, :]])
            detections = decode_outputs(outputs, scale, pad_x, pad_y, orig_w, orig_h)
            pose = 'unknown'
            if detections:
                best = max(detections, key=lambda d: d[4])
                pose = classify_pose(best[5], best[6], best[:4])

            # 生成标注帧
            bgr_ann = bgr.copy()
            draw_detections(bgr_ann, detections, pose, sitting_total, standing_total)
            yuv = cv2.cvtColor(bgr_ann, cv2.COLOR_BGR2YUV_I420)
            y = yuv[:orig_h]
            u = yuv[orig_h:orig_h+orig_h//4].reshape(orig_h//2, orig_w//2)
            v = yuv[orig_h+orig_h//4:].reshape(orig_h//2, orig_w//2)
            uv = np.empty((orig_h//2, orig_w), dtype=np.uint8)
            uv[:, 0::2] = u; uv[:, 1::2] = v
            nv12_out = np.ascontiguousarray(np.vstack([y, uv]))

            result_queue.put((bytes(nv12_out), pose))

        except Exception as e:
            print(f'[InferProc] error: {e}', flush=True)
            result_queue.put((None, 'unknown'))

        infer_count += 1
        if infer_count % 100 == 0:
            _trim()
            rss = _rss_mb()
            if rss > max_rss_mb:
                print(f'[InferProc:{os.getpid()}] RSS {rss:.0f}MB > {max_rss_mb}MB, exiting for restart', flush=True)
                result_queue.put(None)  # 通知主进程即将退出
                break

    rknn.release()
    print(f'[InferProc:{os.getpid()}] exited after {infer_count} inferences', flush=True)


# ═══════════════════════════════════════════════════════════════
# 主进程：采集 + 推流 + 子进程管理
# ═══════════════════════════════════════════════════════════════
class PoseMonitor:
    def __init__(self, model_path, camera='/dev/video0', log_file='pose_log.csv',
                 width=1280, height=720, rtsp_url=None, infer_every=3,
                 infer_max_rss=400):
        self.model_path   = model_path
        self.camera       = camera
        self.log_file     = log_file
        self.width        = width
        self.height       = height
        self.rtsp_url     = rtsp_url
        self.infer_every  = infer_every
        self.infer_max_rss = infer_max_rss  # 子进程 RSS 阈值

        # 姿态状态
        self.current_pose   = None
        self.pose_start_time = None
        self.session_stats  = {'sitting': 0.0, 'standing': 0.0, 'unknown': 0.0}
        self.last_log_time  = time.time()

        # 推理子进程通信队列
        self.frame_queue    = mp.Queue(maxsize=1)
        self.result_queue   = mp.Queue(maxsize=2)
        self._infer_proc    = None

        # 缓存
        self.last_annotated_nv12 = None
        self.last_pose      = 'unknown'
        self.frame_count    = 0
        self.frame_pts      = 0

        # GStreamer
        self.pipeline    = None
        self.out_pipeline = None
        self.appsrc      = None
        self.loop        = None

    # ── CSV 日志 ──────────────────────────────────────────────
    def init_csv(self):
        write_header = not os.path.exists(self.log_file)
        self.csv_file   = open(self.log_file, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if write_header:
            self.csv_writer.writerow(
                ['timestamp','pose','duration_sec','sitting_total','standing_total'])
        self.csv_file.flush()

    def log_pose_change(self, old_pose, duration):
        self.session_stats[old_pose] += duration
        if old_pose == 'unknown':
            return
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([
            ts, old_pose, f'{duration:.1f}',
            f'{self.session_stats["sitting"]:.1f}',
            f'{self.session_stats["standing"]:.1f}'])
        self.csv_file.flush()
        print(f'[{ts}] {old_pose} lasted {duration:.1f}s | '
              f'Sitting: {self.session_stats["sitting"]:.0f}s, '
              f'Standing: {self.session_stats["standing"]:.0f}s')

    def update_state(self, new_pose):
        now = time.time()
        if new_pose != self.current_pose:
            if self.current_pose is not None and self.pose_start_time is not None:
                duration = now - self.pose_start_time
                if duration > 1.0:
                    self.log_pose_change(self.current_pose, duration)
            self.current_pose    = new_pose
            self.pose_start_time = now
        if now - self.last_log_time > 30:
            dur = now - (self.pose_start_time or now)
            print(f'[Status] Current: {self.current_pose} ({dur:.0f}s) | '
                  f'Sitting: {self.session_stats["sitting"]:.0f}s, '
                  f'Standing: {self.session_stats["standing"]:.0f}s | '
                  f'MainRSS: {_rss_mb():.0f}MB')
            self.last_log_time = now

    # ── 推理子进程管理 ────────────────────────────────────────
    def _start_infer_proc(self):
        p = mp.Process(
            target=_inference_process,
            args=(self.model_path, self.frame_queue, self.result_queue,
                  self.width, self.height, self.infer_max_rss),
            daemon=True)
        p.start()
        self._infer_proc = p
        print(f'[Main] inference subprocess started PID={p.pid}')

    def _result_reader(self):
        """后台线程：读取子进程结果，检测子进程退出并重启"""
        while True:
            try:
                item = self.result_queue.get(timeout=5)
            except Exception:
                # 超时：检查子进程是否还活着
                if self._infer_proc and not self._infer_proc.is_alive():
                    print('[Main] inference subprocess died unexpectedly, restarting...')
                    self._start_infer_proc()
                continue

            if item is None:
                # 子进程主动退出（RSS超限），重启
                if self._infer_proc:
                    self._infer_proc.join(timeout=3)
                print('[Main] restarting inference subprocess...')
                # 清空 frame_queue 避免旧帧积压
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except: break
                self._start_infer_proc()
                continue

            nv12_bytes, pose = item
            if nv12_bytes is not None:
                arr = np.frombuffer(nv12_bytes, dtype=np.uint8).reshape(
                    self.height * 3 // 2, self.width)
                self.last_annotated_nv12 = arr
            self.last_pose = pose
            self.update_state(pose)

    # ── GStreamer 回调（主进程，30fps）────────────────────────
    def on_new_sample(self, appsink):
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK

        buf  = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value('width'), caps.get_value('height')

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            nv12 = bytes(mapinfo.data)

            self.frame_count += 1
            if self.frame_count % self.infer_every == 0:
                try:
                    now = time.time()
                    ongoing = now - (self.pose_start_time or now)
                    sit = self.session_stats['sitting'] + (ongoing if self.current_pose == 'sitting' else 0)
                    std = self.session_stats['standing'] + (ongoing if self.current_pose == 'standing' else 0)
                    self.frame_queue.put_nowait((nv12, w, h, sit, std))
                except Exception:
                    pass  # 队列满，跳过本帧

            if self.appsrc is not None:
                ann = self.last_annotated_nv12
                data = bytes(ann) if ann is not None else nv12
                gbuf = Gst.Buffer.new_wrapped(data)
                gbuf.pts      = self.frame_pts
                gbuf.duration = Gst.util_uint64_scale(1, Gst.SECOND, 30)
                self.frame_pts += gbuf.duration
                self.appsrc.emit('push-buffer', gbuf)

        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    # ── GStreamer 管线 ────────────────────────────────────────
    def build_pipeline(self):
        in_str = (
            f'v4l2src device={self.camera} ! '
            f'image/jpeg,width={self.width},height={self.height} ! '
            f'mppjpegdec ! video/x-raw,format=NV12 ! '
            f'appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false'
        )
        print(f'Input:  {in_str}')
        self.pipeline = Gst.parse_launch(in_str)
        self.pipeline.get_by_name('sink').connect('new-sample', self.on_new_sample)

        if self.rtsp_url:
            caps = (f'video/x-raw,format=NV12,'
                    f'width={self.width},height={self.height},framerate=30/1')
            out_str = (
                f'appsrc name=src is-live=true format=time block=false caps={caps} ! '
                f'mpph264enc ! rtspclientsink protocols=tcp location={self.rtsp_url}'
            )
            print(f'Output: {out_str}')
            self.out_pipeline = Gst.parse_launch(out_str)
            self.appsrc = self.out_pipeline.get_by_name('src')

    # ── 主入口 ────────────────────────────────────────────────
    def run(self):
        self.init_csv()
        self.build_pipeline()

        # 启动推理子进程
        self._start_infer_proc()

        # 启动结果读取线程
        t = threading.Thread(target=self._result_reader, daemon=True)
        t.start()

        if self.out_pipeline:
            self.out_pipeline.set_state(Gst.State.PLAYING)
        self.pipeline.set_state(Gst.State.PLAYING)
        print(f'Running. infer_every={self.infer_every}, '
              f'infer_max_rss={self.infer_max_rss}MB')

        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print('\nStopping...')
        finally:
            # 停止子进程
            try: self.frame_queue.put_nowait(None)
            except: pass
            if self._infer_proc:
                self._infer_proc.join(timeout=5)
                if self._infer_proc.is_alive():
                    self._infer_proc.terminate()

            # 记录最后一段
            if self.current_pose and self.pose_start_time:
                duration = time.time() - self.pose_start_time
                if duration > 1.0:
                    self.log_pose_change(self.current_pose, duration)

            self.pipeline.set_state(Gst.State.NULL)
            if self.out_pipeline:
                self.out_pipeline.set_state(Gst.State.NULL)
            self.csv_file.close()

            print('\n=== Session Summary ===')
            for k in ('sitting', 'standing', 'unknown'):
                v = self.session_stats[k]
                print(f'{k:10s}: {v:.0f}s ({v/60:.1f} min)')
            print(f'Log: {self.log_file}')


if __name__ == '__main__':
    mp.set_start_method('spawn')   # RKNN 需要干净的子进程环境

    parser = argparse.ArgumentParser(description='Pose Monitor - Sitting/Standing Detection')
    parser.add_argument('--model', default='models/yolov8_pose/yolov8n-pose-rk3588-fp.rknn')
    parser.add_argument('--camera', default='/dev/video0')
    parser.add_argument('--log', default='pose_log.csv')
    parser.add_argument('--width',  type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--infer-every', type=int, default=3,
                        help='每 N 帧推理一次 (默认3，约10fps)')
    parser.add_argument('--infer-max-rss', type=int, default=400,
                        help='推理子进程 RSS 阈值 MB，超出则重启子进程 (默认400)')
    parser.add_argument('--stream', default=None, metavar='RTSP_URL')
    args = parser.parse_args()

    monitor = PoseMonitor(
        model_path    = args.model,
        camera        = args.camera,
        log_file      = args.log,
        width         = args.width,
        height        = args.height,
        rtsp_url      = args.stream,
        infer_every   = args.infer_every,
        infer_max_rss = args.infer_max_rss,
    )
    monitor.run()
