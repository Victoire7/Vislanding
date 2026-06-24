# Vislanding 发布文件清单

## `main` 分支（核心）

| 路径 | 说明 |
|------|------|
| `README.md` `LICENSE` `requirements.txt` | 项目说明与依赖 |
| `mono/` | 推理模型、配置、工具 |
| `deployment/jetson/` | ONNX / TensorRT 部署 |
| `safe_pre_demo.py` `safe_pre_mask.py` | 推理 demo |
| `eval_safe.py` `eval_safe_multi.py` | 评测 |
| `hubconf.py` | Torch Hub 入口 |
| `training/mono/` | 训练（**不含** ablation / semi / 0226 / 2026_new 实验配置） |
| `training/kitti_json_files/` | 数据集 json 生成 |
| `training/data_server_info/` | 数据路径模板 |
| `final_model/README.md` `weight/README.md` | 权重说明 |
| `media/screenshots/` | 示意图（不含大体积 gif） |
| `docs/` | 发布文档 |

## `test` 分支（`main` + 辅助）

在 `main` 基础上额外包含：

| 路径 | 说明 |
|------|------|
| `training/mono/configs/ablation/` | 消融实验配置 |
| `training/mono/configs/semi/` | 半监督实验配置 |
| `training/mono/configs/0226/` `2026_new/` | 阶段性实验配置 |
| `training/scripts/` | 数据预处理、法向生成等脚本 |
| `iros_exp/` | 额外实验代码 |
| `eval_semantic/` | 语义评测相关 |
| `cal_area_demo*.py` `vis_*.py` `make_video.py` | 可视化与面积计算 demo |
| `finetune.sh` `data/` `onnx/` | 微调脚本、demo 数据、ONNX 工具 |
| `media/gifs/` | 演示动图 |

## 不上传

```
*.pth *.onnx *.plan
training/work_dirs/
bench_models/
.idea/
Eval_*.txt
```
