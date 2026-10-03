# Pose Monitor — 坐站检测直播系统

基于 Orange Pi 5（RK3588）的实时姿态监控系统。使用 NPU 推理 YOLOv8-pose 模型，检测画面中的人是坐着还是站立，记录时长，并通过 RTSP 直播带标注的视频流。

## 硬件要求

- **Orange Pi 5**（RK3588 SoC）
- USB 摄像头（支持 NV12 格式输出，1280×720@15fps）
- 网络连接（局域网访问直播）

## 系统架构

```
/dev/video0  NV12 1280×720 @ 15fps
    │
    ▼  v4l2src (GStreamer 输入管线)
appsink
    │
    ▼
on_new_sample (主进程, 15fps)
    │
    ├── 每 5 帧 ──────────────────────────────────────► frame_queue
    │                                                  (mp.Queue, 1)
    │                                                       │
    └── 每帧                                                ▼
         │                                       推理子进程 (独立进程, 3fps)
         ▼                                         RGA NV12 → BGR
    RGA NV12 → BGR   (硬件 DMA)                   letterbox → 320×320
    draw_detections  (cv2 overlay)                 YOLOv8-pose NPU ~35ms
    RGA BGR → NV12   (硬件 DMA)                   decode (1,56,2100)
    appsrc push-buffer                             classify_pose
         │                                              │
         ▼                                              ▼ result_queue
    mpph264enc (MPP 硬件编码)              _result_reader 线程
         │                                  last_pose / last_detections
         ▼
    rtspclientsink → mediamtx :8554/cam
         │
         ▼
    rtsp://192.168.x.x:8554/cam   (VLC / 手机)
    http://192.168.x.x:8889/cam   (WebRTC 浏览器)
    http://192.168.x.x:8888/cam   (HLS)
```

**关键设计：**
- **推理子进程隔离**：RKNN 运行时在独立进程中，RSS 超限时自动重启，主进程推流不中断
- **异步 overlay**：主进程用最新缓存的推理结果绘制每一帧，推流帧率与推理延迟解耦
- **NV12 直通**：摄像头直接输出 NV12，省去 JPEG 解码，无 mppjpegdec 参与
- **RGA 硬件色彩转换**：NV12↔BGR 走 Rockchip RGA 硬件，CPU 占用极低

## 一、系统依赖安装

### 1.1 GStreamer 基础包

```bash
sudo apt update
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    python3-gi \
    python3-gst-1.0
```

### 1.2 gst-rockchip（mpph264enc 插件）

提供 `mpph264enc` 硬件 H.264 编码器（Orange Pi 官方镜像通常已预装）：

```bash
gst-inspect-1.0 mpph264enc
```

若插件缺失，从源码编译：

```bash
sudo apt install -y meson ninja-build libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libdrm-dev

git clone https://github.com/JeffyCN/rockchip_mirrors.git -b gstreamer-rockchip --depth=1
cd rockchip_mirrors
meson setup build
ninja -C build
sudo ninja -C build install
sudo ldconfig
```

### 1.3 librga（RGA 硬件色彩转换）

```bash
sudo apt install -y librga-dev
```

编译 Python 调用接口：

```bash
cd /home/chun/Develop/pose
gcc -O2 -shared -fPIC -o rga_cvt.so rga_cvt.c -lrga
```

若编译成功，启动时日志会出现 `[RGA] 硬件色彩转换已启用`。

## 二、Python 依赖

```bash
cd /home/chun/Develop/pose
python3 -m venv .venv
source .venv/bin/activate

# 基础依赖（numpy 必须 <2，cv2 与 numpy 2.x 不兼容）
pip install opencv-python "numpy<2"

# RKNN Lite 运行时（Python 3.10）
wget https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 重新固定 numpy<2（rknn-toolkit-lite2 会拉取 numpy 2.x，必须覆盖）
pip install "numpy<2" --force-reinstall
```

**注意**：每次安装/升级 rknn-toolkit-lite2 后都需要重新执行最后一行。

## 三、更新 RKNN Runtime 库

```bash
# 备份旧版本
cp /usr/lib/librknnrt.so ~/librknnrt.so.bak

# 下载 2.3.2
wget -O /tmp/librknnrt_new.so \
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

# 验证版本（应输出 2.3.2）
strings /tmp/librknnrt_new.so | grep -E "^[0-9]+\.[0-9]+\.[0-9]+"

sudo cp /tmp/librknnrt_new.so /usr/lib/librknnrt.so
```

librknnrt 版本需与 rknn-toolkit-lite2 版本匹配（本项目均为 2.3.2）。

## 四、RKNN 模型

> 转换命令、输出格式定义、解码推导过程和 fp/i8 量化对比详见 [RKNN_NOTES.md](RKNN_NOTES.md)。

使用 320×320 输入的 YOLOv8n-pose 模型，推理速度比 640×640 快约 1.7 倍（35ms vs 60ms）。

```bash
mkdir -p models/yolov8_pose
# 将 yolov8n-pose-320-rk3588-fp.rknn 放入此目录
```

模型规格：
- 输入：`(1, 320, 320, 3)` RGB NHWC
- 输出：单张量 `(1, 56, 2100)`
  - `[0:4, :]`：bbox (cx, cy, w, h)，模型输入坐标系
  - `[4, :]`：置信度 logit
  - `[5::3, :] / [6::3, :] / [7::3, :]`：17 关键点 x / y / visibility logit
- 锚点：2100 = 40×40 + 20×20 + 10×10（三尺度，stride 8/16/32）

## 五、MediaMTX（流媒体服务器）

```bash
mkdir -p mediamtx
wget https://github.com/bluenviron/mediamtx/releases/download/v1.17.1/mediamtx_v1.17.1_linux_arm64v8.tar.gz
tar xf mediamtx_v1.17.1_linux_arm64v8.tar.gz -C mediamtx/
```

`mediamtx/mediamtx.yml` 最小配置：

```yaml
rtsp: true
rtspAddress: :8554

hls: true
hlsAddress: :8888

webrtc: true
webrtcAddress: :8889
webrtcLocalUDPAddress: :8189

paths:
  cam:
```

## 六、开机自启（用户级 systemd 服务）

服务文件已放在 `~/.config/systemd/user/`，无需 sudo。

```bash
# 首次配置
systemctl --user daemon-reload
systemctl --user enable --now mediamtx pose_monitor

# 开启 linger：让用户服务在未登录时也能开机自启
loginctl enable-linger chun
```

日常管理：

```bash
# 查看状态 / 日志
systemctl --user status pose_monitor
journalctl --user -u pose_monitor -f

# 重启 / 停止
systemctl --user restart pose_monitor
systemctl --user stop mediamtx
```

## 七、手动运行

```bash
cd /home/chun/Develop/pose
source .venv/bin/activate

# 先启动 mediamtx
./mediamtx/mediamtx mediamtx/mediamtx.yml &

# 启动 pose_monitor
python3 pose_monitor.py \
    --model models/yolov8_pose/yolov8n-pose-320-rk3588-fp.rknn \
    --width 1280 \
    --height 720 \
    --stream rtsp://127.0.0.1:8554/cam
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `models/yolov8_pose/yolov8n-pose-320-rk3588-fp.rknn` | RKNN 模型路径 |
| `--camera` | `/dev/video0` | 摄像头设备节点 |
| `--log` | `pose_log.csv` | CSV 日志输出路径 |
| `--width` | `1280` | 采集分辨率宽 |
| `--height` | `720` | 采集分辨率高 |
| `--infer-every` | `5` | 每 N 帧推理一次（15fps 下约 3fps 推理）|
| `--infer-max-rss` | `400` | 推理子进程 RSS 阈值（MB），超出则重启 |
| `--sit-remind` | `30` | 连续坐超过 N 分钟显示久坐提醒 |
| `--stream` | 无 | RTSP 推流地址 |

### 查看直播

```
rtsp://192.168.x.x:8554/cam      # RTSP（VLC / 手机）
http://192.168.x.x:8889/cam      # WebRTC（浏览器，低延迟）
http://192.168.x.x:8888/cam      # HLS（兼容性好，延迟约 3s）
```

## 八、输出文件

### pose_log.csv

```csv
timestamp,pose,duration_sec,sitting_total,standing_total
2026-04-15 10:00:00,sitting,120.3,120.3,0.0
2026-04-15 10:02:00,standing,30.1,120.3,30.1
```

- 每次姿态切换且持续 > 1 秒时记录一行
- `unknown`（画面中无人）不写入 CSV，但计入内部统计
- 程序正常退出时打印 session 汇总

## 九、性能参考（1280×720，RK3588）

| 环节 | 实测 | 说明 |
|------|------|------|
| 采集帧率 | 15fps | 摄像头 NV12 原生输出 |
| 推理帧率 | ~3fps | infer_every=5，NPU 约 35ms/帧 |
| 推流帧率 | 15fps | 与推理解耦，不受 NPU 延迟影响 |
| CPU 总占用 | ~12% | 8 核合计（含 overlay 绘制） |
| 色彩转换 | ~0% CPU | RGA 硬件 DMA |

## 十、硬件加速使用情况

| 处理环节 | 加速方式 | 组件 |
|----------|---------|------|
| NV12↔BGR 色彩转换 | 硬件 RGA | `rga_cvt.so` + librga |
| H.264 编码 | 硬件 VPU | `mpph264enc`（MPP）|
| 姿态推理（YOLOv8-pose）| 硬件 NPU | `RKNNLite`（NPU_CORE_AUTO）|
| 图像缩放（letterbox）| CPU NEON | `cv2.resize` |

## 十一、软件版本

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04 LTS (Jammy) |
| 内核 | 6.1.x-rockchip-rk3588 |
| librknnrt | 2.3.2 |
| rknn-toolkit-lite2 | 2.3.2 |
| MediaMTX | v1.17.1 |
| GStreamer | 1.20.x |
| OpenCV | 4.x |
| numpy | 1.26.x（必须 <2）|

## 十二、文件结构

```
pose/
├── pose_monitor.py              # 主程序
├── rga_cvt.c                    # RGA 色彩转换 C 源码
├── rga_cvt.so                   # 编译产物（运行时加载）
├── pose_log.csv                 # 姿态日志（运行时生成）
├── README.md
├── models/
│   └── yolov8_pose/
│       └── yolov8n-pose-320-rk3588-fp.rknn   # 320×320 单输出模型
├── mediamtx/
│   ├── mediamtx                 # 二进制
│   └── mediamtx.yml
└── ~/.config/systemd/user/
    ├── mediamtx.service
    └── pose_monitor.service
```

## 十三、常见问题

**Q: 摄像头不支持 NV12 输出**
```bash
# 检查摄像头支持的格式
v4l2-ctl --device=/dev/video0 --list-formats-ext
# 若没有 NV12，但有 MJPEG，需修改 _build_in_pipeline() 改回 mppjpegdec 路径
```

**Q: `[RGA] 加载失败` / 色彩转换走软件路径**
```bash
# 确认 rga_cvt.so 已编译
ls -la rga_cvt.so
# 确认 librga 已安装
ldconfig -p | grep librga
```

**Q: 推理子进程频繁重启**
```
RSS 超过 --infer-max-rss 阈值时子进程主动退出并重启，主进程推流不中断。
若重启过于频繁，适当调大阈值：--infer-max-rss 600
```

**Q: 浏览器看不到画面**
```
先确认 mediamtx 已启动：systemctl --user status mediamtx
再确认 pose_monitor 已推流：journalctl --user -u pose_monitor -f
检查防火墙是否放通 8554/TCP、8889/TCP+UDP。
```

**Q: ImportError: cv2 找不到 _ARRAY_API**
```bash
pip install "numpy<2" --force-reinstall
```

## 十四、Hermes 微信抓帧命令

用户级 `media_delivery` 插件提供 `/rtsp` 命令，从默认流 `rtsp://localhost:8554/cam` 抓取一帧并发送到微信；也支持 `/rtsp rtsp://其他地址`。插件源码和安装步骤见 [Hermes 集成说明](integrations/hermes/README.md)。
