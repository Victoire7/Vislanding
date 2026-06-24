# 发布到 GitHub

仓库：`git@github.com:Victoire7/Vislanding.git`

| 分支 | 用途 |
|------|------|
| `main` | 核心代码（推理 / 训练 / 部署） |
| `test` | 核心 + 实验配置与辅助脚本 |

## 一键发布

```bash
cd <repo_root>

# 推送 main（核心）
bash scripts/publish_vislanding.sh main

# 推送 test（含辅助文件）
bash scripts/publish_vislanding.sh test

# 仅暂存不推送
bash scripts/publish_vislanding.sh main --no-push
```

## 手动命令

```bash
git remote add vislanding git@github.com:Victoire7/Vislanding.git  # 首次

# 删除旧发布分支（如存在）
git push vislanding --delete release/vislanding

# 强制更新 main / test（确认无误后）
git push -u vislanding main --force
git push -u vislanding test --force
```

## 权重

大文件请用 **GitHub Releases** 分发，不要 `git add *.pth`。
