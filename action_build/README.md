# MotionBERT 动作识别实验

YOLO26-pose 的 COCO 17 点可以接到 MotionBERT 的 NTU60 动作分类模型。
这是独立的实验流程，候选动作没有接入微信事件，也没有常驻动作识别进程。
正常 pose 服务仍推流约 5fps、推理约 1fps。

## 数据与模型约定

- pose 每个成功推理结果将当前检测到的人物写到 `/run/pose/skeleton.json`。
  文件原子替换，只保留最新关键点、置信度、P/T 编号和时间戳，不保存图片。
- 按 `monitor_pid + track_id + profile_id` 分离人物；轨迹消失、服务重启、
  超过 2.5 秒未更新或图像尺寸改变时清空序列。
- 使用约 8 秒滑窗，至少 8 个真实检测结果；每帧至少 6 个可信关键点。
  重采样到 32 或 243 帧仅用于匹配模型输入，不增加实际的时间信息。
- 预处理对齐官方 `ActionDataset` 的坐标转换、置信度排列、第二人物零填充，
  以及 `scale_range_test=[2,2]`。其中官方动作加载器转换坐标后仍保留
  COCO 置信度顺序，本实验为兼容现有权重保留该约定。
- 每次仅评估一名人物，第二人按官方单人约定填充；双人交互类别没有验证。
- NTU60 是封闭的 60 类动作，没有“静止/未知”类别。“坐下”是动作，
  不等于持续坐着。分类分数不是动作真实发生的概率，不能直接作为告警。

## 模型来源

- 源码：[Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT)
- 验证版本：`705d3a95354db8bdb696b3492e47a3b5537174ff`
- 权重：[官方作者发布的 NTU60 x-sub 动作分类模型](https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin)
- SHA256：`a932fa9d36cd4ae5e9e56d6003713b05e235ff9a52de2356656dd2637e050481`
- 标签按 [NTU RGB+D 官方类别表](https://github.com/shahroudy/NTURGB-D#action-classes)
  的 A1–A60 顺序，脚本中的 ID 从 0 开始。

完整分类模型约 230MiB；MotionBERT-Lite 没有现成匹配的 NTU60 分类头，
不能直接套用完整版分类权重。原始权重、ONNX 和私人采样数据不提交仓库。

## 导出与验证

Python 3.12 的独立环境中安装 `requirements-build.txt`，下载官方源码与权重后：

```bash
python action_build/export_motionbert.py /path/to/MotionBERT /path/to/best_epoch.bin \
  /path/to/motionbert-ntu60-32.onnx --frames 32
```

脚本严格加载所有权重，比较 PyTorch 与 ONNX 输出并记录模型来源和误差。
`--sample` 可以传入预处理后的 `[1,2,T,17,3]` 真实序列 `.npy`。
`--frames 243` 可比较官方训练序列长度。

RKNN 输入使用四维张量，导出时加 `--rknn-friendly`：它合并时间注意力的
head/关节维度，避免 rank-5 MatMul；会额外核对该变换与原网络的输出一致。

```bash
python action_build/export_motionbert.py /path/to/MotionBERT /path/to/best_epoch.bin \
  /path/to/motionbert-ntu60-32-rk.onnx --frames 32 --rknn-friendly
python action_build/convert_motionbert.py /path/to/motionbert-ntu60-32-rk.onnx \
  /path/to/motionbert-ntu60-32-fp16.rknn
```

转换需要 Linux RKNN Toolkit2 2.5.0。禁用了已经遇到错误的 SDPA、
`squeeze_nd_mul` 和 `swap_reshape_exsoftmax13` 优化规则。
转换成功也必须在设备上验证真实输入与 ONNX 的输出误差，才能判断是否可用。

## 设备采样和 CPU 实验

```bash
python3 action_build/collect_skeletons.py /run/pose/action-trial.jsonl --seconds 45
/root/.cache/pose-motionbert/venv/bin/python action_build/run_motionbert.py \
  /root/.cache/pose-motionbert/motionbert-ntu60-32.onnx \
  /run/pose/action-trial.jsonl --output /run/pose/action-trial-report.json
```

设备端只需要 `requirements.txt`，安装在独立环境中。报告包含前五个候选、
真实采样帧数/帧率、窗口时长、推理耗时和进程峰值 RSS。
`--max-clips 1 --runs 3` 可以只测试一个窗口。

```bash
python -m unittest discover -s action_build -p 'test_*.py'
```

后续要验证准确度，需采集带动作标签、足够帧率的真实序列，并加入静止、
遮挡等负样本。现有 ReID embedding 不能替代动作关键点序列。

## 2026-10-14 实测结论

采集了 44 个实际推理快照，预处理与官方代码的最大差为 `9.5e-8`。
真实输入上的 PyTorch/ONNX logits 最大差为 `7.63e-5`，设备和本地
ONNX 也得到一致的候选类别。

| RK3576 CPU 配置 | 32 帧单次推理 | 进程峰值 RSS |
|---|---:|---:|
| 两线程 | 平均 4.92 秒 | 492.6MiB |
| 四个大核、四线程 | 3.53 秒 | 486.7MiB |

32 帧和官方 243 帧在相同序列上产生不同候选，实际输入仅约 1fps，
目前没有人工动作标签，不能据此给出准确率。预训练权重仍需验证场景适配。

RKNN 2.5.0 转换已进入后端编译阶段，但超过了独立编译进程的
`MemoryMax=1800M` 上限，被 cgroup OOM 终止，未生成可用的 RKNN 文件。
此结果不能证明模型无法在 NPU 上运行；下一次应在内存更充足的 Linux
构建机编译，再复制到 RK3576 测试。CPU 路径已跑通，但暂不常驻启用。

汇总数据见 [experiment_20261014.json](experiment_20261014.json)。
