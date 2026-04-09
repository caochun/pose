"""
姿态监控：检测画面中的人是坐着还是站立，并记录时长
使用 GStreamer appsink + YOLOv8-pose RKNN NPU 推理

优化：
- 异步推理线程：推流帧率不受 NPU 推理延迟影响
- NV12 输出：省去 BGR→NV12 软件 videoconvert
- 推理限频：每 N 帧推理一次，其余帧复用上次结果
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
import threading
import queue
import argparse
from datetime import datetime
from rknnlite.api import RKNNLite

Gst.init(None)

# ─── YOLOv8-pose 关键点索引 (COCO 17点) ───
KP_NOSE       = 0
KP_LEFT_EYE   = 1
KP_RIGHT_EYE  = 2
KP_LEFT_EAR   = 3
KP_RIGHT_EAR  = 4
KP_LEFT_SHLDR = 5
KP_RIGHT_SHLDR= 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW= 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST= 10
KP_LEFT_HIP   = 11
KP_RIGHT_HIP  = 12
KP_LEFT_KNEE  = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE= 16

INPUT_SIZE = 640
NMS_THRESH = 0.45
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
    keypoints_all = outputs[3]  # (1, 17, 3, 8400)
    boxes = []
    strides = [8, 16, 32]
    grid_sizes = [80, 40, 20]

    for i, (stride, gs) in enumerate(zip(strides, grid_sizes)):
        feat = outputs[i].reshape(1, 65, -1)
        bbox = feat[0, :64, :]
        conf = sigmoid(feat[0, 64, :])

        idx_start = sum(g*g for g in grid_sizes[:i])
        kps = keypoints_all[0, :, :, idx_start:idx_start+gs*gs]

        for ci in range(gs * gs):
            if conf[ci] < CONF_THRESH:
                continue
            cx = ci % gs
            cy = ci // gs

            b = bbox[:, ci].reshape(4, 16)
            b = softmax(b, axis=1)
            b = (b * np.arange(16)).sum(axis=1)
            x1 = (cx + 0.5 - b[0]) * stride
            y1 = (cy + 0.5 - b[1]) * stride
            x2 = (cx + 0.5 + b[2]) * stride
            y2 = (cy + 0.5 + b[3]) * stride

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            kp = kps[:, :, ci]
            kp_coords = kp[:, :2].copy()
            kp_coords[:, 0] = (kp_coords[:, 0] - pad_x) / scale
            kp_coords[:, 1] = (kp_coords[:, 1] - pad_y) / scale
            kp_confs = kp[:, 2]

            boxes.append([x1, y1, x2, y2, float(conf[ci]), kp_coords, kp_confs])

    if not boxes:
        return []

    scores = np.array([b[4] for b in boxes])
    indices = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
        scores.tolist(), CONF_THRESH, NMS_THRESH
    )
    if len(indices) == 0:
        return []
    return [boxes[i] for i in indices.flatten()]


def classify_pose(kp_coords, kp_confs, box):
    MIN_CONF = 0.3
    hip_ys, knee_ys = [], []
    if kp_confs[KP_LEFT_HIP]   > MIN_CONF: hip_ys.append(kp_coords[KP_LEFT_HIP][1])
    if kp_confs[KP_RIGHT_HIP]  > MIN_CONF: hip_ys.append(kp_coords[KP_RIGHT_HIP][1])
    if kp_confs[KP_LEFT_KNEE]  > MIN_CONF: knee_ys.append(kp_coords[KP_LEFT_KNEE][1])
    if kp_confs[KP_RIGHT_KNEE] > MIN_CONF: knee_ys.append(kp_coords[KP_RIGHT_KNEE][1])

    if not hip_ys or not knee_ys:
        return 'unknown'

    box_h = box[3] - box[1]
    if box_h < 1:
        return 'unknown'

    vertical_ratio = (np.mean(knee_ys) - np.mean(hip_ys)) / box_h
    return 'standing' if vertical_ratio > 0.15 else 'sitting'


def draw_detections(frame, detections, pose):
    """在 frame 上原地绘制骨骼点和姿态标签"""
    if not detections:
        return
    best = max(detections, key=lambda d: d[4])
    x1, y1, x2, y2 = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    color = (0, 255, 0) if pose == 'standing' else (0, 165, 255) if pose == 'sitting' else (128, 128, 128)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, pose, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    kp_coords, kp_confs = best[5], best[6]
    for ki in range(17):
        if kp_confs[ki] > 0.3:
            cv2.circle(frame, (int(kp_coords[ki][0]), int(kp_coords[ki][1])), 4, (255, 0, 0), -1)


class PoseMonitor:
    def __init__(self, model_path, camera='/dev/video0', log_file='pose_log.csv',
                 width=1280, height=720, rtsp_url=None, infer_every=3):
        self.model_path = model_path
        self.camera = camera
        self.log_file = log_file
        self.width = width
        self.height = height
        self.rtsp_url = rtsp_url
        self.infer_every = infer_every  # 每 N 帧推理一次

        # 状态追踪
        self.current_pose = None
        self.pose_start_time = None
        self.session_stats = {'sitting': 0.0, 'standing': 0.0, 'unknown': 0.0}
        self.last_log_time = time.time()

        # 异步推理
        self.infer_queue = queue.Queue(maxsize=1)
        self.last_detections = None
        self.last_bgr = None
        self.last_annotated_nv12 = None  # 推理线程写好的标注 NV12，回调直接用
        self.frame_count = 0

        self.rknn = None
        self.pipeline = None
        self.out_pipeline = None
        self.appsrc = None
        self.frame_pts = 0
        self.loop = None
        self._infer_thread = None

    def init_rknn(self):
        print(f'Loading RKNN model: {self.model_path}')
        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(self.model_path) != 0:
            raise RuntimeError('Failed to load RKNN model')
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError('Failed to init RKNN runtime')
        print('RKNN model loaded OK')

    def init_csv(self):
        write_header = not os.path.exists(self.log_file)
        self.csv_file = open(self.log_file, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if write_header:
            self.csv_writer.writerow(['timestamp', 'pose', 'duration_sec', 'sitting_total', 'standing_total'])
        self.csv_file.flush()

    def log_pose_change(self, old_pose, duration):
        self.session_stats[old_pose] += duration
        if old_pose == 'unknown':
            return
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([
            ts, old_pose, f'{duration:.1f}',
            f'{self.session_stats["sitting"]:.1f}',
            f'{self.session_stats["standing"]:.1f}'
        ])
        self.csv_file.flush()
        print(f'[{ts}] {old_pose} lasted {duration:.1f}s | '
              f'Total sitting: {self.session_stats["sitting"]:.0f}s, '
              f'standing: {self.session_stats["standing"]:.0f}s')

    def update_state(self, new_pose):
        now = time.time()
        if new_pose != self.current_pose:
            if self.current_pose is not None and self.pose_start_time is not None:
                duration = now - self.pose_start_time
                if duration > 1.0:
                    self.log_pose_change(self.current_pose, duration)
            self.current_pose = new_pose
            self.pose_start_time = now

        if now - self.last_log_time > 30:
            current_duration = now - (self.pose_start_time or now)
            print(f'[Status] Current: {self.current_pose} ({current_duration:.0f}s) | '
                  f'Sitting: {self.session_stats["sitting"]:.0f}s, '
                  f'Standing: {self.session_stats["standing"]:.0f}s')
            self.last_log_time = now

    # ─── 异步推理线程（输入 BGR，NV12→BGR 在此完成）──────────────
    def _inference_worker(self):
        while True:
            item = self.infer_queue.get()
            if item is None:
                break
            nv12_frame, (orig_w, orig_h) = item
            try:
                # NV12→BGR（只在推理帧做，1/infer_every 频率）
                bgr = cv2.cvtColor(nv12_frame, cv2.COLOR_YUV2BGR_NV12)
                inp, scale, pad_x, pad_y = letterbox(bgr, INPUT_SIZE)
                inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                outputs = self.rknn.inference(inputs=[inp_rgb[np.newaxis, :]])
                detections = decode_outputs(outputs, scale, pad_x, pad_y, orig_w, orig_h)
                pose = 'unknown'
                if detections:
                    best = max(detections, key=lambda d: d[4])
                    pose = classify_pose(best[5], best[6], best[:4])
                # 把 BGR 和检测结果缓存，供回调线程绘制
                self.last_bgr = bgr
                self.last_detections = (detections, pose)
                self.update_state(pose)

                # 推理线程做 BGR→NV12（仅 1/infer_every 频率），回调直接复用
                bgr_ann = bgr.copy()
                draw_detections(bgr_ann, detections, pose)
                yuv = cv2.cvtColor(bgr_ann, cv2.COLOR_BGR2YUV_I420)
                y = yuv[:orig_h]
                u = yuv[orig_h:orig_h + orig_h // 4].reshape(orig_h // 2, orig_w // 2)
                v = yuv[orig_h + orig_h // 4:].reshape(orig_h // 2, orig_w // 2)
                uv = np.empty((orig_h // 2, orig_w), dtype=np.uint8)
                uv[:, 0::2] = u
                uv[:, 1::2] = v
                self.last_annotated_nv12 = np.ascontiguousarray(np.vstack([y, uv]))
            except Exception as e:
                print(f'[Infer error] {e}')
            finally:
                self.infer_queue.task_done()

    # ─── GStreamer 回调（NV12 快速路径）────────────────────────────
    def on_new_sample(self, appsink):
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value('width')
        h = caps.get_value('height')

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            # NV12 帧：Y(H×W) + UV(H/2×W)，共 H*W*3//2 字节
            nv12 = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h * 3 // 2, w).copy()

            # 每 infer_every 帧投递到推理线程
            self.frame_count += 1
            if self.frame_count % self.infer_every == 0:
                try:
                    self.infer_queue.put_nowait((nv12.copy(), (w, h)))
                except queue.Full:
                    pass

            # 推送到编码器
            if self.appsrc is not None:
                ann = self.last_annotated_nv12
                data = ann.tobytes() if ann is not None else bytes(mapinfo.data)

                gbuf = Gst.Buffer.new_wrapped(data)
                gbuf.pts = self.frame_pts
                gbuf.duration = Gst.util_uint64_scale(1, Gst.SECOND, 30)
                self.frame_pts += gbuf.duration
                self.appsrc.emit('push-buffer', gbuf)

            if args.show:
                bgr_show = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                if self.last_detections:
                    draw_detections(bgr_show, *self.last_detections)
                cv2.imshow('Pose Monitor', bgr_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.loop.quit()
        except Exception as e:
            print(f'[Callback error] {e}')
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def build_pipeline(self):
        # 输入：NV12 直出，省掉 videoconvert；30fps 全速给 appsink
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
            # NV12 直接给 mpph264enc，省掉 videoconvert
            caps = (f'video/x-raw,format=NV12,'
                    f'width={self.width},height={self.height},framerate=30/1')
            out_str = (
                f'appsrc name=src is-live=true format=time block=false caps={caps} ! '
                f'mpph264enc ! '
                f'rtspclientsink protocols=tcp location={self.rtsp_url}'
            )
            print(f'Output: {out_str}')
            self.out_pipeline = Gst.parse_launch(out_str)
            self.appsrc = self.out_pipeline.get_by_name('src')

    def run(self):
        self.init_rknn()
        self.init_csv()
        self.build_pipeline()

        # 启动推理线程
        self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._infer_thread.start()

        if self.out_pipeline:
            self.out_pipeline.set_state(Gst.State.PLAYING)
        self.pipeline.set_state(Gst.State.PLAYING)
        print(f'Running. infer_every={self.infer_every}. Press Ctrl+C to stop.')

        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print('\nStopping...')
        finally:
            # 停止推理线程
            self.infer_queue.put(None)
            self._infer_thread.join(timeout=3)

            if self.current_pose and self.pose_start_time:
                duration = time.time() - self.pose_start_time
                if duration > 1.0:
                    self.log_pose_change(self.current_pose, duration)

            self.pipeline.set_state(Gst.State.NULL)
            if self.out_pipeline:
                self.out_pipeline.set_state(Gst.State.NULL)
            self.rknn.release()
            self.csv_file.close()
            if args.show:
                cv2.destroyAllWindows()

            print('\n=== Session Summary ===')
            print(f'Sitting   : {self.session_stats["sitting"]:.0f}s ({self.session_stats["sitting"]/60:.1f} min)')
            print(f'Standing  : {self.session_stats["standing"]:.0f}s ({self.session_stats["standing"]/60:.1f} min)')
            print(f'No person : {self.session_stats["unknown"]:.0f}s ({self.session_stats["unknown"]/60:.1f} min)')
            print(f'Log saved to: {self.log_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Monitor - Sitting/Standing Detection')
    parser.add_argument('--model', default='/home/orangepi/Develop/models/yolov8_pose/yolov8n-pose-rk3588-fp.rknn')
    parser.add_argument('--camera', default='/dev/video0')
    parser.add_argument('--log', default='pose_log.csv')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--infer-every', type=int, default=3,
                        help='每 N 帧推理一次 (默认3，即约 10fps 推理)')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--stream', default=None, metavar='RTSP_URL')
    args = parser.parse_args()

    monitor = PoseMonitor(
        model_path=args.model,
        camera=args.camera,
        log_file=args.log,
        width=args.width,
        height=args.height,
        rtsp_url=args.stream,
        infer_every=args.infer_every,
    )
    monitor.run()
