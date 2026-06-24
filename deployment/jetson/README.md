# Vislanding — Jetson 部署

把 `RAFTDepthNormalSafeDPT5`（ViT-small + DINOv2-reg + RAFT-T 步解码器 + Safe 头）
导出到 ONNX，在 **Jetson** 上以 TensorRT 推理。本目录附带一键脚本，可以
**测 4 种配置（FP32 / FP16 × T=1 / T=4）的 FPS**。

> 关于 RAFT 的 T 步迭代：解码器里 `for itr in range(self.iters):` 会在 ONNX 导出时被
> tracer 展开成 T 段串联子图。**T 是烧进图里的常量**，运行时不能改；要换 T 必须
> 重新导出 ONNX。

cfg / ckpt 默认值与 `safe_pre_demo.py` 一致：

| 项目 | 值 |
|---|---|
| cfg  | `mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py` |
| ckpt | `final_model/student_step00004400_86.04.pth` |
| 输入 | `616 x 1064`（ImageNet 归一化，归一化已封装进 ONNX） |

可在 `deployment/jetson/defaults.env` 修改，也能通过环境变量临时覆盖。

---

## 目录布局

```
deployment/jetson/
├── defaults.env               # cfg/ckpt/输入尺寸/benchmark 网格的默认值
├── pipeline.py                # 预/后处理 (ONNX & TRT 共用)
├── safe_metric3d_export.py    # PyTorch -> ONNX (含全部 export-time patch)
├── count_model_params.py      # 统计参数量（encoder/decoder 分项）
├── count_model_flops.py       # 统计 FLOPs（随 RAFT iters T 变化）
├── safe_metric3d_fp16.py      # ONNX FP32 -> ONNX FP16（FPS bench 不需要）
├── verify_onnx.py             # PyTorch vs ONNXRuntime 数值对齐
├── infer_onnxruntime.py       # ORT 推理 demo
├── infer_tensorrt.py          # 纯 TensorRT 推理 demo
├── build_trt_engine.sh        # 普通构建 .plan（带可视化推理时用）
├── prepare_bench.sh           # ★ 主机端：一键导出 T=1, T=4 的 FP32 ONNX
├── bench_fps.py               # ★ Jetson 端：trtexec 合成输入 FPS（4 个组合）
├── bench_fps_real.py          # ★ Jetson 端：真实图片端到端 FPS（4 个组合）
├── requirements_export.txt    # 主机依赖
└── requirements_jetson.txt    # Jetson 依赖
```

---

## 统计原始模型参数量

统计 **PyTorch 原始网络**（`DepthModel` → `depth_model` → encoder + decoder），
不含 ONNX 导出 wrapper。在已安装训练环境的主机上：

```bash
cd <repo_root>
python deployment/jetson/count_model_params.py
python deployment/jetson/count_model_params.py --detail --json params_report.json
```

默认 cfg/ckpt 与 `defaults.env` 一致；`--no-ckpt` 只建图不加载权重（参数量相同）。
`decode_head.iters`（T）只改变 RAFT 前向步数，**不改变参数量**。

### 统计 FLOPs

```bash
pip install thop       # 主机上推荐；fvcore 对 ViT+RAFT 易 trace 失败
python deployment/jetson/count_model_flops.py
python deployment/jetson/count_model_flops.py --iters 1 4 --json flops_report.json
```

默认输入 `616×1064`；FLOPs **随 T 增大**（与参数量不同）。`decoder/T` 列为粗估每步迭代开销。

---

## 一键 FPS Benchmark（FP32/FP16 × T=1/T=4）

### Step 1：主机端 — 导出 2 份 FP32 ONNX

```bash
cd <repo_root>
pip install -r deployment/jetson/requirements_export.txt

./deployment/jetson/prepare_bench.sh ./bench_models/
```

读取 `defaults.env` 里的 cfg/ckpt，导出 `safe_metric3d_T1.onnx` 和
`safe_metric3d_T4.onnx`。要换 ckpt：

```bash
CKPT=/path/to/another.pth ./deployment/jetson/prepare_bench.sh ./bench_models/
```

> **不需要预先转 FP16 ONNX** —— trtexec 在 Jetson 端用 `--fp16` 直接量化更快、更稳。

### Step 2：把 ONNX + 工具脚本拷到 Jetson

```bash
scp -r ./bench_models                jetson@<ip>:~/safe_metric3d/
scp -r deployment/jetson             jetson@<ip>:~/safe_metric3d_tools/
```

### Step 3：Jetson 端 — 一键构建 4 个引擎 + 跑 FPS

```bash
ssh jetson@<ip>
cd ~/safe_metric3d_tools
pip install -r requirements_jetson.txt

# （强烈建议）先开高性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

python3 bench_fps.py ~/safe_metric3d/
```

`bench_fps.py` 默认遍历 `iters=[1,4]` × `precisions=[fp32, fp16]`：

1. 调用 `trtexec` 构建 `safe_metric3d_T{T}_{prec}.plan`（FP16 自动加 `--fp16`）；  
2. 用 `--useCudaGraph --noDataTransfers` 跑 200 次纯 GPU 计算计时；  
3. 解析 mean / median GPU 计算时间，换算 FPS；  
4. 输出 markdown 表 + CSV（保存到 `bench_models/bench_fps.csv` 和 `.md`）。

输出示例：

```
| precision | T | GPU compute mean (ms) | GPU compute median (ms) | FPS (mean) | FPS (median) | trtexec throughput (qps) | engine |
|---|---|---|---|---|---|---|---|
| fp32 | 1 | 28.42 | 28.31 | 35.18 | 35.33 | 35.20 | safe_metric3d_T1_fp32.plan |
| fp16 | 1 | 10.83 | 10.79 | 92.34 | 92.68 | 92.30 | safe_metric3d_T1_fp16.plan |
| fp32 | 4 | 45.07 | 44.92 | 22.19 | 22.26 | 22.20 | safe_metric3d_T4_fp32.plan |
| fp16 | 4 | 17.21 | 17.18 | 58.11 | 58.21 | 58.10 | safe_metric3d_T4_fp16.plan |
```

> 数字仅作示例；实际数值取决于具体的 Jetson 型号 / 功率档 / cuDNN 与 TRT 版本。

#### 选项

```bash
# 只跑 fp16
python3 bench_fps.py ~/safe_metric3d/ --precisions fp16

# 跑更多 T
python3 bench_fps.py ~/safe_metric3d/ --iters 1 2 4 8

# 已经有 .plan，跳过重新构建直接计时
python3 bench_fps.py ~/safe_metric3d/                   # 默认 reuse 现有 plan
python3 bench_fps.py ~/safe_metric3d/ --rebuild         # 强制重建

# 修改 timing 参数
python3 bench_fps.py ~/safe_metric3d/ --warmup-ms 5000 --iterations 500

# 指定 trtexec 路径（默认自动找 /usr/src/tensorrt/bin/trtexec）
python3 bench_fps.py ~/safe_metric3d/ --trtexec /usr/src/tensorrt/bin/trtexec
```

---

## 真实图片端到端 FPS（推荐）

`bench_fps.py` 用 trtexec 的合成输入只能测**纯 GPU 计算**；想看真实业务里的
FPS（含 `cv2` decode + resize + pad + numpy 转换 + H2D + GPU + D2H + 可选后处理），
用 `bench_fps_real.py`：

```bash
# Jetson 端，假设 bench_models 里已有 safe_metric3d_T{1,4}.onnx
# 第二个参数是放真实图片的目录（jpg/jpeg/png/bmp/webp 任意混合）
python3 bench_fps_real.py ~/safe_metric3d/ ~/datasets/test_images/

# 想把后处理时间也计进 FPS：
python3 bench_fps_real.py ~/safe_metric3d/ ~/datasets/test_images/ --postprocess

# 图片少时跑多遍以拿到稳定统计
python3 bench_fps_real.py ~/safe_metric3d/ ~/datasets/test_images/ --repeat 5

# 只测一部分子集
python3 bench_fps_real.py ~/safe_metric3d/ ~/datasets/test_images/ --max-images 30
```

输出按 `(precision, T)` 4 个组合各一行，分项报告 preproc / GPU / postproc / e2e
的 mean / median / p99，并给出 `FPS (e2e mean)` 和 `FPS (gpu only)` 两个指标，
方便看出预处理瓶颈：

```
| precision | T | preproc mean (ms) | gpu mean (ms) | e2e mean (ms) | e2e median (ms) | e2e p99 (ms) | FPS (e2e mean) | FPS (gpu only) |
|---|---|---|---|---|---|---|---|---|
| fp32 | 1 | 6.21 | 28.40 | 35.10 | 34.95 | 38.20 | 28.49 | 35.21 |
| fp16 | 1 | 6.18 | 10.85 | 17.55 | 17.40 | 19.80 | 56.98 | 92.17 |
| fp32 | 4 | 6.23 | 45.06 | 51.85 | 51.70 | 55.30 | 19.29 | 22.19 |
| fp16 | 4 | 6.20 | 17.18 | 23.92 | 23.80 | 26.10 | 41.81 | 58.21 |
```

> 数字仅作示例；实际取决于 Jetson 型号、功耗档、cuDNN/TRT 版本、图片分辨率。

脚本会自动**先构建缺失的 `.plan`**（与 `bench_fps.py` 共用同样的命名约定，
所以两个脚本可以混用同一份 `bench_models/`），加 `--rebuild` 强制重建。

输出 `bench_models/bench_fps_real.csv` + `.md`。

---

## 需要拷到 Jetson 的最小文件清单

| 必须 / 可选 | 文件 | 说明 |
|---|---|---|
| 必须 | `bench_models/safe_metric3d_T1.onnx`, `_T4.onnx` | 2 份 ONNX |
| 必须 | `deployment/jetson/bench_fps.py` | 合成输入 FPS（纯 GPU 计算） |
| 必须 (真实图片测试时) | `deployment/jetson/bench_fps_real.py` | 端到端 FPS |
| 必须 (真实图片测试时) | `deployment/jetson/infer_tensorrt.py` | 提供 `TrtRunner` |
| 必须 (真实图片测试时) | `deployment/jetson/pipeline.py` | 提供 `preprocess` / `postprocess_*` |
| 必须 | `deployment/jetson/defaults.env` | 默认值（可选 source） |
| 必须 (Jetson) | `numpy` + `opencv-python` + `pycuda`（或 `cuda-python`） | 见 `requirements_jetson.txt` |
| 可选 | `deployment/jetson/build_trt_engine.sh` | 想单独构建引擎时用 |
| 可选 | `deployment/jetson/requirements_jetson.txt` | 列出 Python 包 |

> bench_fps.py 本身只依赖 `trtexec`（JetPack 自带）+ Python 标准库；  
> bench_fps_real.py 额外需要 `numpy` / `opencv-python` / `pycuda` 或 `cuda-python`。  
> 两者都**不需要 PyTorch / mono/ 源码 / `.pth` 权重**。

---

## 其它常用工作流

仅当你想做**带预处理 / 可视化的端到端推理**时才需要：

```bash
# 主机端：把 FP32 ONNX 转成 FP16 ONNX（仅用于 ORT，不用于 trtexec FPS bench）
python deployment/jetson/safe_metric3d_fp16.py \
  --input  bench_models/safe_metric3d_T4.onnx \
  --output bench_models/safe_metric3d_T4_fp16.onnx --simplify

# Jetson 端：纯 TensorRT 推理 + 保存可视化
python deployment/jetson/infer_tensorrt.py \
  --engine bench_models/safe_metric3d_T4_fp16.plan \
  --image  /path/to/foo.jpg \
  --intrinsic 1500 1500 960 540 \
  --out-dir ./out

# 主机端：数值对齐校验（PyTorch vs ONNX）
python deployment/jetson/verify_onnx.py \
  --cfg  mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py \
  --onnx bench_models/safe_metric3d_T4.onnx
```

---

## 注意事项 / 常见坑

- **引擎不可跨 GPU 复用**：`.plan` 与具体 SM 绑定，必须在最终目标 Jetson 上构建。  
- **第一次 trtexec 构建很慢**（T=4 大约 5–15 分钟，取决于 Orin 型号）。bench_fps.py
  缓存 `.plan`，下次直接复用计时。  
- **测之前一定 `sudo jetson_clocks`**，否则 DVFS 抖动让数据没法比。  
- **T=1 必有精度损失**：模型按 T=4 训练，跑 T=1 等于只跑 1 步 RAFT 修正；
  benchmark 的同时建议跑一次 `verify_onnx.py` 看 depth/safe logits 的 `max|diff|`。  
- **trtexec 的 `--noDataTransfers` 只影响计时**，不影响构建出来的 plan；
  实际部署时 H2D/D2H 拷贝开销另计（一般 < 2 ms）。  
- **MemPool**：默认 4 GB 工作区。Orin Nano 8GB 上若 OOM，加 `--workspace-mb 2048`。  

---

## 参考

- `safe_pre_demo.py` — PyTorch 端 demo，cfg/ckpt 与本目录默认值一致。  
- `mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py` — 模型 / 解码器 cfg（含 `iters`）。  
- `mono/model/decode_heads/RAFTDepthNormalSafeDPTDecoder5_bestbak.py` — Safe 解码器实现。  
