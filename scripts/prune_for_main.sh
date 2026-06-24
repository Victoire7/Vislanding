#!/usr/bin/env bash
# Remove auxiliary paths from the index for the lean main branch.
set -euo pipefail
cd "$(dirname "$0")/.."
GIT=/usr/bin/git

REMOVE_PATHS=(
  data/
  onnx/
  test.sh
  test_kitti.sh
  test_nyu.sh
  test_vit.sh
  requirements_v1.txt
  requirements_v2.txt
  docs/METRIC3D_README.md
  media/gifs/
  finetune.sh
  hubconf2.py
  cal_area_demo.py
  cal_area_demo_real.py
  vis_gt.py
  vis_3d_bar.py
  vis_miou_bar.py
  make_video.py
  iros_exp/
  eval_semantic/
  training/scripts/
  training/mono/configs/ablation/
  training/mono/configs/semi/
  training/mono/configs/0226/
  training/mono/configs/2026_new/
)

for p in "${REMOVE_PATHS[@]}"; do
  $GIT rm -r --cached "$p" 2>/dev/null || true
done

# keep only essential RAFTDecoder configs on main (wilduav / safe / kitti baseline)
while IFS= read -r f; do
  case "$f" in
    training/mono/configs/RAFTDecoder/vit.raft5.small.wilduav*.py) ;;
    training/mono/configs/RAFTDecoder/vit.raft5.small.kitti.py) ;;
    training/mono/configs/RAFTDecoder/vit.raft5.small.ddos.py) ;;
    training/mono/configs/RAFTDecoder/vit.raft5.large.kitti.py) ;;
    *) $GIT rm --cached "$f" 2>/dev/null || true ;;
  esac
done < <($GIT ls-files 'training/mono/configs/RAFTDecoder/*.py' 2>/dev/null || true)

echo "[prune] main tree trimmed"
