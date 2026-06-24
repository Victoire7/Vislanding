# Safe 学生模型 Checkpoint

本目录用于存放 Vislanding Safe 学生模型，**不随 Git 仓库分发**。

## 推荐文件

| 文件名 | 说明 |
|--------|------|
| `student_step00004400_86.04.pth` | 默认部署 / demo 权重 |

## 使用

```bash
export CKPT=final_model/student_step00004400_86.04.pth
python safe_pre_demo.py

CKPT=$CKPT bash deployment/jetson/prepare_bench.sh ./bench_models/
```

## 下载

> 在 GitHub **Releases** 或团队网盘填写下载链接。

预训练 backbone 见 `weight/README.md`。
