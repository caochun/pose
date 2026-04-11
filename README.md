# Pose Monitor — 坐站检测直播系统

基于 Orange Pi 5（RK3588）的实时姿态监控系统。使用 NPU 推理 YOLOv8-pose 模型，检测画面中的人是坐着还是站立，记录时长，并通过 WebRTC 直播带标注的视频流。

## 硬件要求

- **Orange Pi 5**（RK3588 SoC）
- USB 摄像头（支持 MJPEG 1280×720@30fps）
- 网络连接（局域网访问直播）

## 系统架构

```
摄像头 (MJPEG 30fps)
    │
    ▼  [硬件 VPU] mppjpegdec
    │  JPEG → NV12
    ▼
appsink (Python GStreamer 回调)
    │
    ├── 每3帧 → 推理队列
    │              │
    │         推理线程 (独立)
    │              │  NV12→BGR (cv2 NEON)
    │              │  letterbox resize 640×640
    │              │  YOLOv8-pose → [硬件 NPU]
    │              │  解码输出 / 坐站分类
    │              │  绘制标注 (cv2)
    │              │  BGR→NV12 (cv2 NEON)
    │              └→ last_annotated_nv12 + CSV日志
    │
    ▼ last_annotated_nv12
appsrc
    │
    ▼  [硬件 VPU] mpph264enc
    │  NV12 → H.264
    ▼
rtspclientsink → MediaMTX
    ├── WebRTC  :8889  ← 浏览器访问
    └── HLS     :8888
```

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

### 1.2 Rockchip MPP（硬件编解码库）

MPP 通常已随 Orange Pi 官方镜像预装。验证：

```bash
ls /usr/lib/aarch64-linux-gnu/librockchip_mpp.so*
```

若缺失，从源码编译：

```bash
git clone https://github.com/rockchip-linux/mpp.git
cd mpp
cmake -B build -DRKPLATFORM=ON -DHAVE_DRM=ON
cmake --build build -j4
sudo cmake --install build
```

### 1.3 gst-rockchip（mppjpegdec / mpph264enc 插件）

这是关键插件，提供 `mppjpegdec`（硬件 JPEG 解码）和 `mpph264enc`（硬件 H.264 编码）。

**Orange Pi 官方镜像**通常已预装此插件，先验证：

```bash
gst-inspect-1.0 mppjpegdec
gst-inspect-1.0 mpph264enc
```

若两条命令均有输出，跳过本节。若插件缺失，从源码编译：

```bash
sudo apt install -y meson ninja-build libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libdrm-dev

# 原 rockchip-linux/gst-rockchip 已移除，使用 JeffyCN 镜像
git clone https://github.com/JeffyCN/rockchip_mirrors.git -b gstreamer-rockchip --depth=1
cd rockchip_mirrors
meson setup build
ninja -C build
sudo ninja -C build install
sudo ldconfig
```

> 也可使用 [BoxCloudIRL/gstreamer-rockchip](https://github.com/BoxCloudIRL/gstreamer-rockchip)，该分支针对 RK3588 流媒体场景有额外修复。

## 二、Python 依赖

```bash
# 基础依赖（numpy 必须 <2，cv2 与 numpy 2.x 不兼容）
pip install opencv-python "numpy<2"

# RKNN Lite 运行时（Python API）
# 根据 Python 版本选择 wheel（此处为 Python 3.10）
wget https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# wheel 文件名必须完整，否则 pip 报错
pip install rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 重新固定 numpy<2（rknn-toolkit-lite2 会拉取 numpy 2.x，必须覆盖）
pip install "numpy<2" --force-reinstall
```

**注意**：每次安装/升级 rknn-toolkit-lite2 后都需要重新执行最后一行。

## 三、更新 RKNN Runtime 库

librknnrt.so 是 NPU 推理的核心运行时，系统预装版本可能较旧，需手动更新到最新版：

```bash
# 备份旧版本
cp /usr/lib/librknnrt.so ~/librknnrt.so.bak

# 查看当前版本
strings /usr/lib/librknnrt.so | grep -E "^[0-9]+\.[0-9]+\.[0-9]+"

# 下载 2.3.2（最新，2025年）
wget -O /tmp/librknnrt_new.so \
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

# 验证版本
strings /tmp/librknnrt_new.so | grep -E "^[0-9]+\.[0-9]+\.[0-9]+"
# 应输出: 2.3.2 (...)

# 替换
sudo cp /tmp/librknnrt_new.so /usr/lib/librknnrt.so
```

**说明**：librknnrt 版本需与 rknn-toolkit-lite2 版本匹配（本项目均为 2.3.2）。

## 四、下载 RKNN 模型

```bash
mkdir -p models/yolov8_pose

# 从 rknn_model_zoo 下载预转换模型（FP 浮点版，精度优先）
wget -O models/yolov8_pose/yolov8n-pose-rk3588-fp.rknn \
  https://raw.githubusercontent.com/airockchip/rknn_model_zoo/main/models/CV/object_detection/yolo/RKNN/yolov8n-pose-rk3588-fp.rknn
```

模型说明：
- `yolov8n`：nano 版本，速度快，适合实时推理
- `fp`：浮点精度（非量化），精度更高
- 输入：640×640 RGB，`NCHW` 格式
- 输出：4 个张量
  - `(1,65,80,80)`、`(1,65,40,40)`、`(1,65,20,20)`：三尺度 bbox + 置信度
  - `(1,17,3,8400)`：17 个关键点（COCO 格式，每点含 x/y/置信度）

若需要从 ONNX 自行转换，需在 x86 开发机上安装 rknn-toolkit2（非 lite2）。

## 五、MediaMTX（流媒体服务器）

```bash
mkdir -p mediamtx
wget https://github.com/bluenviron/mediamtx/releases/download/v1.17.1/mediamtx_v1.17.1_linux_arm64v8.tar.gz
tar xf mediamtx_v1.17.1_linux_arm64v8.tar.gz -C mediamtx/
```

配置文件 `mediamtx/mediamtx.yml` 最小配置：

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
    # pose_monitor 通过 rtspclientsink 推流到此路径
    # 无需额外配置，允许匿名推流
```

## 六、运行

### 6.1 启动 MediaMTX

```bash
cd /home/orangepi/Develop/mediamtx
./mediamtx mediamtx.yml > /tmp/mediamtx.log 2>&1 &
```

### 6.2 启动 Pose Monitor

```bash
cd /home/orangepi/Develop
python3 pose_monitor.py \
    --model models/yolov8_pose/yolov8n-pose-rk3588-fp.rknn \
    --width 1280 \
    --height 720 \
    --stream rtsp://127.0.0.1:8554/cam
```

等待输出 `RKNN model loaded OK` 和 `Running.` 后，流媒体即可访问。

### 6.3 查看直播

浏览器打开（替换为板子的局域网 IP）：

```
http://192.168.x.x:8889/cam      # WebRTC（低延迟，推荐）
http://192.168.x.x:8888/cam      # HLS（兼容性好，延迟约 3s）
```

### 6.4 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `models/yolov8_pose/yolov8n-pose-rk3588-fp.rknn` | RKNN 模型路径 |
| `--camera` | `/dev/video0` | 摄像头设备节点 |
| `--log` | `pose_log.csv` | CSV 日志输出路径 |
| `--width` | `1280` | 摄像头分辨率宽 |
| `--height` | `720` | 摄像头分辨率高 |
| `--infer-every` | `3` | 每 N 帧推理一次（默认 ~10fps 推理）|
| `--infer-max-rss` | `400` | 推理子进程 RSS 阈值（MB），超出则重启子进程（主进程推流不中断）|
| `--stream` | 无 | RTSP 推流地址（不指定则不推流）|

## 七、输出文件

### pose_log.csv

```csv
timestamp,pose,duration_sec,sitting_total,standing_total
2026-04-09 13:48:54,sitting,34.8,34.8,0.0
2026-04-09 13:56:31,standing,27.6,0.0,27.6
```

- 每次姿态切换且持续时间 > 1 秒时记录一行
- `unknown`（画面中无人）不写入 CSV，但计入内部统计
- 程序正常退出（Ctrl+C）时打印 session 汇总

## 八、性能参考（1280×720，RK3588）

| 线程 / 进程 | CPU 占用 | 说明 |
|------------|---------|------|
| GStreamer 主回调 | ~4% | NV12 取帧 + 推 appsrc |
| 推理线程 | ~44% | NPU 推理 + 色彩转换 + 绘制 |
| mppjpegdec | ~18% | 硬件 JPEG 解码（内核态） |
| mpph264enc | ~4% | 硬件 H.264 编码（内核态） |
| **系统合计** | **~15%** | 8 核合计，空闲约 85% |

NPU 推理约 10fps（infer_every=3），直播流维持 30fps 不受推理延迟影响。

## 九、硬件加速使用情况

| 处理环节 | 加速方式 | 组件 |
|----------|---------|------|
| JPEG 解码 | 硬件 VPU | `mppjpegdec` |
| H.264 编码 | 硬件 VPU | `mpph264enc` |
| 姿态推理（YOLOv8-pose）| 硬件 NPU | `RKNNLite`（NPU_CORE_AUTO，自动分配3核）|
| 色彩转换（NV12↔BGR）| CPU NEON | `cv2.cvtColor` |
| 图像缩放（letterbox）| CPU NEON | `cv2.resize` |

> **关于 RGA**：Rockchip RGA 2D 加速器理论上支持色彩空间转换。实测在 1280×720 分辨率下，每次调用的内核 ioctl 开销约 1.5ms，而 OpenCV NEON 用户态计算仅约 0.5ms，RGA 反而更慢。RGA 在 4K 分辨率或完整 DMA-buf 零拷贝链路下才有优势。本项目保持使用 cv2。

## 十、软件版本

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04 LTS (Jammy) |
| 内核 | 6.1.99-rockchip-rk3588 |
| RKNPU 内核驱动 | 0.9.8 (builtin) |
| librknnrt | 2.3.2 |
| rknn-toolkit-lite2 | 2.3.2 |
| MediaMTX | v1.17.1 |
| GStreamer | 1.20.x |
| OpenCV | 4.x |
| numpy | 1.26.4（必须 <2）|

## 十一、文件结构

```
Develop/
├── pose_monitor.py              # 主程序
├── pose_log.csv                 # 姿态日志（运行时自动生成）
├── README.md                    # 本文件
├── models/
│   └── yolov8_pose/
│       └── yolov8n-pose-rk3588-fp.rknn
└── mediamtx/
    ├── mediamtx                 # 二进制可执行文件
    └── mediamtx.yml             # 配置文件
```

`rga_utils.py`（如存在）为 RGA 硬件加速实验代码，生产环境不需要。

## 十二、常见问题

**Q: 摄像头无法打开 / pipeline 报错**
```
检查设备节点：ls /dev/video*
检查摄像头支持的格式：v4l2-ctl --device=/dev/video0 --list-formats-ext
确认摄像头支持 MJPEG 1280×720
```

**Q: mppjpegdec 报 "invalid output format"**
```
mppjpegdec 仅支持 NV12 输出，不支持 BGR/RGB/BGRA。
确认 pipeline 中 format=NV12。
```

**Q: 浏览器看不到画面**
```
先启动 MediaMTX，再启动 pose_monitor.py。
等待 "Running." 输出后再刷新浏览器。
检查防火墙是否放通 8889/UDP 和 8889/TCP。
```

**Q: ImportError: cv2 找不到 _ARRAY_API**
```
numpy 版本不兼容，执行：
pip install "numpy<2" --force-reinstall
```

**Q: RKNN 推理报错 "need 4dims input"**
```
inference 输入需要增加 batch 维度：
inp_rgb[np.newaxis, :]  # shape: (1, 640, 640, 3)
```
