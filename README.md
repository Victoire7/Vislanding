# Vislanding

无人机视觉着陆感知：从单张 RGB 图像同时预测 **度量深度**、**表面法向** 与 **可着陆区域（Safe）分割**，支持 **ONNX / TensorRT** 在 Jetson 边缘端部署。

默认输入分辨率：**616 × 1064**（ViT-small）。

---

## 功能

| 模块 | 说明 |
|------|------|
| Metric depth | 度量深度（ViT-small + RAFT-DPT） |
| Surface normal | 稠密法向 |
| Safe landing | 可着陆区域分割（logits / prob / mask） |
| Jetson | ONNX 导出、FP16 TensorRT、T=1/4 FPS benchmark |
| Training | WildUAV-Safe 监督 / 半监督训练 |

---

## 安装

```bash
conda create -n vislanding python=3.8 -y
conda activate vislanding
pip install torch torchvision  # 按本机 CUDA 选择 wheel
pip install -r requirements.txt
```

---

## 权重

权重不随仓库分发，见：

- `weight/README.md` — ViT-small 预训练 backbone
- `final_model/README.md` — Safe 学生模型 checkpoint

---

## 推理

```bash
python safe_pre_demo.py
python eval_safe.py
```

配置：`mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py`

---

## 训练

```bash
cd training/mono/tools
python train_safe.py --config ../../mono/configs/final/<config>.py
```

数据集路径：`training/mono/configs/_base_/datasets/safe_wilduav.py`  
生成标注列表：`training/kitti_json_files/generate_safe_wild_json.py`

---

## Jetson 部署

详见 [deployment/jetson/README.md](deployment/jetson/README.md)。

```bash
CKPT=final_model/student_step00004400_86.04.pth \
  bash deployment/jetson/prepare_bench.sh ./bench_models/

python deployment/jetson/bench_fps_real.py ./bench_models /path/to/images
```

---

## 仓库分支

| 分支 | 内容 |
|------|------|
| `main` | 核心推理 / 训练 / 部署代码 |
| `test` | `main` + 实验配置、辅助脚本、可视化与数据处理工具 |

发布命令：`bash scripts/publish_vislanding.sh main` 或 `bash scripts/publish_vislanding.sh test`

---

## 目录结构

```
├── mono/                 # 推理模型与配置
├── training/mono/        # 训练代码
├── deployment/jetson/    # ONNX / TensorRT
├── safe_pre_demo.py
├── eval_safe.py
└── docs/
```

清单：[docs/RELEASE_MANIFEST.md](docs/RELEASE_MANIFEST.md)
