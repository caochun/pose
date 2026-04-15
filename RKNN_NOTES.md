# RKNN 模型转换与推理解码说明

记录 YOLOv8-pose → RKNN 转换过程、输出格式推导和量化选型，供日后更新模型时参考。

---

## 一、模型转换

### 1.1 前置说明

RKNN 转换必须在 **x86 Linux** 机器上完成（rknn-toolkit2 不支持 ARM）。
需要使用 **airockchip/ultralytics** 分支，不能用官方 ultralytics，因为官方导出的 ONNX 结构与 RKNN 转换工具不兼容（头部算子不同）。

### 1.2 环境准备（x86 机器）

```bash
# Python 3.8~3.11 均可，推荐 3.10
pip install rknn-toolkit2==2.3.2

# 使用 airockchip fork，而非官方 ultralytics
pip uninstall ultralytics -y
pip install git+https://github.com/airockchip/ultralytics_yolov8.git
```

### 1.3 导出 ONNX

```python
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
model.export(
    format='onnx',
    imgsz=320,       # 目标输入尺寸，本项目使用 320
    opset=19,
    simplify=True,
)
# 生成 yolov8n-pose.onnx
```

> **关键**：不要加 `half=True` 或 `dynamic=True`，静态 shape 转换更稳定。

### 1.4 转换为 RKNN

```python
from rknn.api import RKNN

rknn = RKNN(verbose=False)

rknn.config(
    target_platform='rk3588',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
)

rknn.load_onnx(model='yolov8n-pose.onnx')

# FP 模式（无量化，精度最高）
rknn.build(do_quantization=False)

rknn.export_rknn('yolov8n-pose-320-rk3588-fp.rknn')
rknn.release()
```

### 1.5 验证模型元数据

转换完成后在 x86 上验证：

```python
from rknn.api import RKNN
rknn = RKNN()
rknn.load_rknn('yolov8n-pose-320-rk3588-fp.rknn')
print(rknn.get_sdk_version())
# 确认输出 shape：应为 [(1, 56, 2100)]
```

---

## 二、输出格式定义

### 2.1 张量结构

模型输出单个张量，shape 为 `(1, 56, N)`：

```
N = 40×40 + 20×20 + 10×10 = 1600 + 400 + 100 = 2100 anchors
    │          │           │
  stride=8   stride=16  stride=32   (对应 320×320 输入)

56 个通道含义：
  [0]     cx       — bbox 中心 x，模型输入像素坐标
  [1]     cy       — bbox 中心 y
  [2]     w        — bbox 宽度
  [3]     h        — bbox 高度
  [4]     conf     — 置信度 logit（需 sigmoid）
  [5]     kp0_x    — 第 0 关键点 x（鼻子）
  [6]     kp0_y
  [7]     kp0_v    — 第 0 关键点 visibility logit（需 sigmoid）
  [8]     kp1_x
  [9]     kp1_y
  [10]    kp1_v
  ...
  [53]    kp16_x   — 第 16 关键点 x（右脚踝）
  [54]    kp16_y
  [55]    kp16_v
```

关键点顺序（COCO 17点）：
```
0  鼻子       1  左眼       2  右眼
3  左耳       4  右耳
5  左肩       6  右肩
7  左肘       8  右肘
9  左腕      10  右腕
11 左髋      12  右髋
13 左膝      14  右膝
15 左踝      16  右踝
```

### 2.2 坐标空间

- bbox `(cx, cy, w, h)` 和关键点 `(x, y)` 均在 **模型输入坐标系**（320×320）
- letterbox 缩放后，映射回原图坐标：
  ```python
  x_orig = (x_model - pad_x) / scale
  y_orig = (y_model - pad_y) / scale
  ```

---

## 三、解码逻辑推导

### 3.1 为什么 bbox 是 cx/cy/w/h 而不是 x1/y1/x2/y2

标准 ultralytics 在 `export` 时会将 Detect head 的 `end2end=True`，此时模型内部直接输出 `(x1,y1,x2,y2)`。

airockchip fork 为了兼容 RKNN 量化流程，使用 `end2end=False`，触发以下路径（`ultralytics/nn/modules/head.py`）：

```python
# Detect.decode_bboxes()
def decode_bboxes(self, bboxes, anchors):
    return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)
    #                                       ^^^^^^^^^^^^^^^^
    #                        end2end=False → xywh=True → 输出 cx,cy,w,h
```

`dist2bbox` 在 `xywh=True` 时输出的是中心点格式，已乘以对应 stride，单位是模型输入像素。

### 3.2 输出已解码，无需手动还原 DFL

与 YOLOv8 的原始 ONNX（输出原始 DFL 分布）不同，airockchip 导出的 RKNN 模型在导出时已将 DFL 分布还原为 `ltrb` 再通过 `dist2bbox` 转换为 `xywh`。

因此 Python 端的解码不需要 DFL softmax + 矩阵乘法，直接读取 `o[0:4]` 即为已解码的 bbox。

### 3.3 关键点 visibility 的处理

关键点的第三分量 `kp_v` 是 logit 值，需要 sigmoid：

```python
kp_v = sigmoid(o[7::3, mask])   # shape (17, M)
```

visibility > 0.3 认为关键点可见（`classify_pose` 和 `draw_detections` 使用此阈值）。
keypoints 的 x/y 不需要 sigmoid，直接使用原始值做坐标映射。

---

## 四、fp 与 i8 量化对比

| 指标 | fp（无量化） | i8（INT8 量化）|
|------|------------|--------------|
| 推理时延（RK3588 NPU）| ~35ms | ~20ms |
| 模型文件大小 | 7.9 MB | ~2.2 MB |
| 关键点精度 | 高 | 稍低，细微姿态可能误判 |
| 转换复杂度 | 无需校准集 | 需要约 100 张代表性图片做校准 |
| 推荐场景 | 精度优先 | 推理速度受限时 |

**本项目选择 fp 的原因**：

1. 坐站分类依赖髋关节/膝关节相对位置，精度下降会直接影响分类结果
2. 35ms 推理 + infer_every=5 实现约 3fps 推理已满足需求，无需量化加速
3. fp 模型无需准备校准数据集，维护成本低

若日后需要 i8，转换脚本只需增加：

```python
rknn.build(
    do_quantization=True,
    dataset='calib_images.txt',   # 每行一个图片路径，约 100 张
)
```
