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

# ─── 温度（RK3588 thermal zone） ─────────────────────────────────
_THERMAL = {
    'soc': '/sys/class/thermal/thermal_zone0/temp',
    'npu': '/sys/class/thermal/thermal_zone6/temp',
}

def _temp_c(key):
    try:
        with open(_THERMAL[key]) as f:
            return int(f.read()) / 1000.0
    except Exception:
        return 0.0

def _npu_load():
    try:
        with open('/sys/class/devfreq/fdab0000.npu/load') as f:
            return int(f.read().split('@')[0])
    except Exception:
        return 0

def _ddr_load():
    try:
        with open('/sys/class/devfreq/dmc/load') as f:
            return int(f.read().split('@')[0])
    except Exception:
        return 0

def _gpu_load():
    try:
        with open('/sys/class/devfreq/fb000000.gpu/load') as f:
            return int(f.read().split('@')[0])
    except Exception:
        return 0

# ─── CPU 使用率（/proc/stat 全局） ───────────────────────────────
_cpu_prev = [0, 0]  # [total_ticks, idle_ticks]

def _cpu_pct():
    try:
        with open('/proc/stat') as f:
            vals = list(map(int, f.readline().split()[1:8]))
        idle  = vals[3]
        total = sum(vals)
        dt = total - _cpu_prev[0]
        di = idle  - _cpu_prev[1]
        _cpu_prev[0] = total
        _cpu_prev[1] = idle
        return max(0.0, (1.0 - di / dt) * 100.0) if dt > 0 else 0.0
    except Exception:
        return 0.0


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


def posture_score(kp_coords, kp_confs, box):
    """根据关键点计算坐姿评分 0(差)~100(好)"""
    MIN_CONF = 0.3
    def get(i):
        return kp_coords[i] if kp_confs[i] > MIN_CONF else None
    nose    = get(KP_NOSE)
    l_shldr = get(KP_LEFT_SHLDR);  r_shldr = get(KP_RIGHT_SHLDR)
    l_hip   = get(KP_LEFT_HIP);    r_hip   = get(KP_RIGHT_HIP)
    bh = box[3] - box[1]
    if bh < 10:
        return 100
    deduct = 0
    shldrs = [s for s in [l_shldr, r_shldr] if s is not None]
    hips   = [h for h in [l_hip,   r_hip  ] if h is not None]
    # 1. 低头：鼻子接近肩膀高度（ratio 正常 >0.08）
    if nose is not None and shldrs:
        sy    = float(np.mean([s[1] for s in shldrs]))
        ratio = (sy - nose[1]) / bh
        if ratio < 0.08:
            deduct += min(35, int((0.08 - ratio) / 0.08 * 35))
    # 2. 肩膀不对称
    if l_shldr is not None and r_shldr is not None:
        sw = abs(l_shldr[0] - r_shldr[0])
        sd = abs(l_shldr[1] - r_shldr[1])
        if sw > 0:
            asym = sd / sw
            if asym > 0.10:
                deduct += min(25, int((asym - 0.10) / 0.15 * 25))
    # 3. 躯干横向偏移（肩中心 vs 髋中心）
    if shldrs and hips:
        sx   = float(np.mean([s[0] for s in shldrs]))
        hx   = float(np.mean([h[0] for h in hips]))
        lean = abs(sx - hx) / bh
        if lean > 0.05:
            deduct += min(20, int((lean - 0.05) / 0.10 * 20))
    return max(0, 100 - deduct)


def _fmt_duration(seconds):
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def draw_detections(frame, detections, pose, sitting_total=0.0, standing_total=0.0, stats=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # ── 右上角：合并面板（左列坐站时长，右列性能指标） ───────────
    pad, lh = 10, 30
    # 坐姿评分颜色
    post = stats.get('posture', 100) if stats else 100
    post_col = ((0,255,128) if post >= 80 else
                (0,200,255) if post >= 60 else
                (0,80, 255))

    # 亮度辅助（需在 left_rows 前计算）
    def _lux_col(y):
        if y > 180: return (220, 255, 255)
        if y >  80: return (0,   255, 128)
        if y >  30: return (0,   200, 255)
        return             (120, 120, 120)
    def _lux_label(y, backlit):
        if backlit:  return 'backlit'
        if y > 180:  return 'bright'
        if y >  80:  return 'normal'
        if y >  30:  return 'dim'
        return              'dark'
    ym   = stats.get('y_mean', 128) if stats else 128
    blit = stats.get('backlit', False) if stats else False
    lux_col = (80, 80, 255) if blit else _lux_col(ym)

    # 左列：LUX / SIT / STD / AWAY / CNT / POST / CONF
    left_rows = [
        ('LUX',  _lux_label(ym, blit),                     lux_col),
        ('SIT',  _fmt_duration(sitting_total),             (0, 165, 255)),
        ('STD',  _fmt_duration(standing_total),            (0, 255,   0)),
        ('AWAY', _fmt_duration(stats.get('away', 0) if stats else 0),
                                                           (160, 160, 160)),
        ('CNT',  (f'{stats.get("sit_cnt",0)}s '
                  f'{stats.get("std_cnt",0)}w') if stats else '0s 0w',
                                                           (200, 200, 200)),
        ('POST', f'{post}',                                post_col),
        ('CONF', f'{stats.get("conf", 0.0):.2f}' if stats else '0.00',
                                                           (255, 200, 50)),
    ]
    # 右列：FPS / CPU / RSS / CONF / 温度
    def _tcol(t):  # 温度颜色：绿→黄→红
        if t < 60:  return (0, 255, 128)
        if t < 75:  return (0, 200, 255)
        return (0, 80, 255)
    def _temp_str(t):
        return f'{t:.0f}C'  # OpenCV Hershey 不支持 Unicode，用小圆+C 代替
    def _draw_temp(frame, x, y, t, col):
        # 画数字+"C"，在"C"前用小圆圈模拟度符号
        s = f'{t:.0f}'
        cv2.putText(frame, s, (x, y), font, 0.60, col, 1, cv2.LINE_AA)
        tw = cv2.getTextSize(s, font, 0.60, 1)[0][0]
        cx, cy = x + tw + 4, y - 8
        cv2.circle(frame, (cx, cy), 2, col, 1, cv2.LINE_AA)
        cv2.putText(frame, 'C', (cx + 5, y), font, 0.60, col, 1, cv2.LINE_AA)

    right_rows = [
        ('FPS',  f'{stats.get("fps",  0.0):.1f}',  (0, 255, 128),  None),
        ('CPU',  f'{stats.get("cpu",  0.0):.0f}%', (0, 255, 255),  None),
        ('NPU',  f'{stats.get("npu_pct",  0):.0f}%',  (0, 200, 255),  None),
        ('GPU',  f'{stats.get("gpu_load", 0)}%',      (160, 255, 160),None),
        ('DDR',  f'{stats.get("ddr_load", 0)}%',      (180, 180, 255),None),
        ('LAT',  f'{stats.get("infer_ms", 0):.0f}ms', (200, 200, 200),None),
        ('RSS',  f'{stats.get("rss",  0):.0f}MB',     (200, 200, 200),None),
        ('SOC',  None, _tcol(stats.get("tsoc", 0)), stats.get("tsoc", 0)),
    ] if stats else []

    n_rows  = max(len(left_rows), len(right_rows))
    lk, lv = 54, 90  # 左列 key宽 / val宽（原36/72 的1.5倍）
    lgap    = 10       # 左列 key与val之间的间距
    rk, rv = 54, 90   # 右列 key宽 / val宽
    sep     = 10       # 分隔列宽
    pw = pad + lk + lgap + lv + sep + rk + rv + pad
    ph = pad + n_rows * lh + pad

    rx, ry = w - pw - 8, 8

    # 半透明黑底
    roi = frame[ry:ry+ph, rx:rx+pw]
    cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
    frame[ry:ry+ph, rx:rx+pw] = roi

    # 竖分隔线
    sx = rx + pad + lk + lv + sep // 2
    cv2.line(frame, (sx, ry + 6), (sx, ry + ph - 6), (80, 80, 80), 1)

    # 左列文字
    for j, (key, val, col) in enumerate(left_rows):
        y = ry + pad + (j + 1) * lh - 4
        x0 = rx + pad
        cv2.putText(frame, key, (x0,             y), font, 0.60, (160,160,160), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (x0 + lk + lgap, y), font, 0.60, col,           1, cv2.LINE_AA)

    # 右列文字
    for j, (key, val, col, temp) in enumerate(right_rows):
        y  = ry + pad + (j + 1) * lh - 4
        x0 = rx + pad + lk + lv + sep
        cv2.putText(frame, key, (x0, y), font, 0.60, (160,160,160), 1, cv2.LINE_AA)
        if temp is not None:
            _draw_temp(frame, x0 + rk, y, temp, col)
        else:
            cv2.putText(frame, val, (x0 + rk, y), font, 0.60, col, 1, cv2.LINE_AA)

    # ── 久坐警告横幅 ──────────────────────────────────────────────
    if stats:
        cont = stats.get('cont_sit', 0)
        remind = stats.get('remind', 1800)
        if cont > 0 and cont >= remind:
            mins = int(cont // 60)
            msg  = f'Stand up! Sitting {mins} min'
            (tw, th), _ = cv2.getTextSize(msg, font, 0.85, 2)
            bx = (w - tw) // 2 - 12
            by = h - 52
            roi2 = frame[by:by+th+20, bx:bx+tw+24]
            cv2.addWeighted(np.full_like(roi2, (0, 0, 180)), 0.75,
                            roi2, 0.25, 0, roi2)
            frame[by:by+th+20, bx:bx+tw+24] = roi2
            cv2.putText(frame, msg, (bx+12, by+th+6),
                        font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # ── 检测框 + 关键点 ───────────────────────────────────────────
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

    infer_count = 0   # 每秒重置，用于 FPS 计算
    total_count = 0   # 累计，用于 RSS 检查
    _fps_t0 = time.time()
    _fps_val = 0.0
    _last_infer_t = None   # 上次推理开始时刻，用于计算 NPU 合成利用率
    _npu_pct = 0.0
    _cpu_pct()  # 初始化 prev 计数器
    while True:
        item = frame_queue.get()
        if item is None:
            break

        (nv12_bytes, orig_w, orig_h) = item
        nv12 = np.frombuffer(nv12_bytes, dtype=np.uint8).reshape(orig_h * 3 // 2, orig_w)

        try:
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            inp, scale, pad_x, pad_y = letterbox(bgr, INPUT_SIZE)
            inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            _t0 = time.time()
            outputs = rknn.inference(inputs=[inp_rgb[np.newaxis, :]])
            infer_ms = (time.time() - _t0) * 1000
            # NPU 合成利用率：推理耗时 / 推理间隔
            if _last_infer_t is not None:
                interval_ms = (_t0 - _last_infer_t) * 1000
                if interval_ms > 0:
                    _npu_pct = min(100.0, infer_ms / interval_ms * 100)
            _last_infer_t = _t0
            detections = decode_outputs(outputs, scale, pad_x, pad_y, orig_w, orig_h)
            pose = 'unknown'
            conf = 0.0
            post = 100
            if detections:
                best = max(detections, key=lambda d: d[4])
                pose = classify_pose(best[5], best[6], best[:4])
                conf = best[4]
                if pose in ('sitting', 'standing'):
                    post = posture_score(best[5], best[6], best[:4])

            # 亮度分析（从 Y 分量，每 8 像素采样一次，约 1.4 万个样本）
            y_samp = nv12[:orig_h:8, ::8]
            y_mean = float(y_samp.mean())
            # 逆光：暗区和亮区同时占比高（双峰分布）
            dark_pct   = float(np.mean(y_samp < 40))  * 100
            bright_pct = float(np.mean(y_samp > 210)) * 100
            backlit    = dark_pct > 15 and bright_pct > 15
            del y_samp, nv12  # NV12 input no longer needed; free before alloc

            # 每秒更新一次 FPS / CPU / RSS
            infer_count += 1
            now = time.time()
            elapsed = now - _fps_t0
            if elapsed >= 1.0:
                _fps_val = infer_count / elapsed
                infer_count = 0
                _fps_t0 = now

            stats = {
                'fps':      _fps_val,
                'cpu':      _cpu_pct(),
                'rss':      _rss_mb(),
                'conf':     conf,
                'tsoc':     _temp_c('soc'),
                'npu_pct':  _npu_pct,
                'gpu_load': _gpu_load(),
                'ddr_load': _ddr_load(),
                'infer_ms': infer_ms,
                'y_mean':   y_mean,
                'backlit':  backlit,
                'posture':  post,
            }

            # 序列化检测结果（numpy → list，供主进程绘制）
            det_serial = [
                (float(d[0]), float(d[1]), float(d[2]), float(d[3]),
                 float(d[4]), d[5].tolist(), d[6].tolist())
                for d in detections
            ]

            result_queue.put((pose, stats, det_serial))
            del bgr, inp, inp_rgb

        except Exception as e:
            print(f'[InferProc] error: {e}', flush=True)
            result_queue.put((pose if 'pose' in dir() else 'unknown', {}, []))

        total_count += 1
        if total_count % 30 == 0:
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
                 infer_max_rss=400, sit_remind_secs=1800):
        self.model_path    = model_path
        self.camera        = camera
        self.log_file      = log_file
        self.width         = width
        self.height        = height
        self.rtsp_url      = rtsp_url
        self.infer_every   = infer_every
        self.infer_max_rss = infer_max_rss
        self.sit_remind_secs = sit_remind_secs

        # 姿态状态
        self.current_pose    = None
        self.pose_start_time = None
        self.session_stats   = {'sitting': 0.0, 'standing': 0.0, 'unknown': 0.0}
        self.last_log_time   = time.time()

        # 离座 / 计数 / 久坐
        self.away_start_time      = None   # 开始离座的时刻
        self.away_total           = 0.0    # 累计离座秒数
        self.sit_count            = 0      # 坐下次数
        self.stand_count          = 0      # 站起次数
        self.continuous_sit_start = None   # 当前连续坐下开始时刻

        # 推理子进程通信队列
        self.frame_queue    = mp.Queue(maxsize=1)
        self.result_queue   = mp.Queue(maxsize=2)
        self._infer_proc    = None

        # 缓存（推理结果，由 _result_reader 线程更新）
        self.last_pose        = 'unknown'
        self.last_infer_stats = {}
        self.last_detections  = []
        self.frame_count      = 0
        self.frame_pts        = 0

        # 串流帧率计数
        self._fps_count = 0
        self._fps_t0    = time.time()
        self._fps_val   = 0.0

        # GStreamer
        self.pipeline    = None
        self.out_pipeline = None
        self.appsrc      = None
        self.loop        = None
        self.push_fail   = 0   # appsrc push-buffer 失败计数

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
            old = self.current_pose
            # 结算旧状态
            if old is not None and self.pose_start_time is not None:
                duration = now - self.pose_start_time
                if duration > 1.0:
                    self.log_pose_change(old, duration)
            # 离座统计
            if new_pose == 'unknown':
                self.away_start_time      = now
                self.continuous_sit_start = None   # 离座打断连续坐
            else:
                if old == 'unknown' and self.away_start_time is not None:
                    self.away_total += now - self.away_start_time
                    self.away_start_time = None
            # 坐/站计数
            if new_pose == 'sitting':
                self.sit_count += 1
                self.continuous_sit_start = now
            elif new_pose == 'standing':
                self.stand_count += 1
                self.continuous_sit_start = None
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
                    if self._infer_proc.is_alive():
                        self._infer_proc.terminate()
                        self._infer_proc.join(timeout=2)
                    if self._infer_proc.is_alive():
                        self._infer_proc.kill()
                print('[Main] restarting inference subprocess...')
                # 清空 frame_queue 避免旧帧积压
                while not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except: break
                self._start_infer_proc()
                continue

            pose, infer_stats, det_serial = item
            self.last_pose        = pose
            self.last_infer_stats = infer_stats
            self.last_detections  = det_serial
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
            self.frame_count += 1
            nv12_bytes = bytes(mapinfo.data)

            # 每 infer_every 帧送一帧给推理子进程
            if self.frame_count % self.infer_every == 0:
                try:
                    self.frame_queue.put_nowait((nv12_bytes, w, h))
                except Exception:
                    pass  # 队列满，跳过

            if self.appsrc is not None:
                # 每帧都绘制 overlay（使用最新缓存的推理结果），保证 30fps 流畅
                now     = time.time()
                self._fps_count += 1
                _elapsed = now - self._fps_t0
                if _elapsed >= 1.0:
                    self._fps_val   = self._fps_count / _elapsed
                    self._fps_count = 0
                    self._fps_t0    = now
                ongoing = now - (self.pose_start_time or now)
                sit  = self.session_stats['sitting']  + (ongoing if self.current_pose == 'sitting'  else 0)
                std  = self.session_stats['standing'] + (ongoing if self.current_pose == 'standing' else 0)
                stats = dict(self.last_infer_stats)
                stats['fps'] = self._fps_val
                stats['away']     = self.away_total + ((now - self.away_start_time) if self.away_start_time else 0)
                stats['sit_cnt']  = self.sit_count
                stats['std_cnt']  = self.stand_count
                stats['cont_sit'] = now - self.continuous_sit_start if self.continuous_sit_start else 0.0
                stats['remind']   = self.sit_remind_secs

                nv12 = np.frombuffer(nv12_bytes, dtype=np.uint8).reshape(h * 3 // 2, w)
                bgr  = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                draw_detections(bgr, self.last_detections, self.last_pose, sit, std,
                                stats if stats else None)
                yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
                y_p = yuv[:h]
                u_p = yuv[h:h+h//4].reshape(h//2, w//2)
                v_p = yuv[h+h//4:].reshape(h//2, w//2)
                uv_p = np.empty((h//2, w), dtype=np.uint8)
                uv_p[:, 0::2] = u_p; uv_p[:, 1::2] = v_p
                out_bytes = np.ascontiguousarray(np.vstack([y_p, uv_p])).tobytes()
                del nv12, bgr, yuv, y_p, u_p, v_p, uv_p

                gbuf = Gst.Buffer.new_allocate(None, len(out_bytes), None)
                gbuf.fill(0, out_bytes)
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

            # 结算离座
            if self.away_start_time:
                self.away_total += time.time() - self.away_start_time

            print('\n=== Session Summary ===')
            for k in ('sitting', 'standing', 'unknown'):
                v = self.session_stats[k]
                print(f'{k:10s}: {v:.0f}s ({v/60:.1f} min)')
            total = sum(self.session_stats.values()) or 1
            sit_v = self.session_stats['sitting']
            std_v = self.session_stats['standing']
            print(f'Away      : {self.away_total:.0f}s ({self.away_total/60:.1f} min)')
            print(f'Sit count : {self.sit_count}  |  Stand count: {self.stand_count}')
            if self.sit_count:
                print(f'Avg sit   : {sit_v/self.sit_count:.0f}s')
            if self.stand_count:
                print(f'Avg stand : {std_v/self.stand_count:.0f}s')
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
    parser.add_argument('--sit-remind', type=int, default=30,
                        help='连续坐超过 N 分钟显示久坐提醒 (默认30)')
    parser.add_argument('--stream', default=None, metavar='RTSP_URL')
    args = parser.parse_args()

    monitor = PoseMonitor(
        model_path      = args.model,
        camera          = args.camera,
        log_file        = args.log,
        width           = args.width,
        height          = args.height,
        rtsp_url        = args.stream,
        infer_every     = args.infer_every,
        infer_max_rss   = args.infer_max_rss,
        sit_remind_secs = args.sit_remind * 60,
    )
    monitor.run()
