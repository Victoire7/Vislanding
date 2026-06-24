#!/usr/bin/env bash
# Rebuild main/test with a single initial commit each (no prior history), then force-push.
#
# Usage:
#   bash scripts/fresh_init_publish.sh           # rebuild + push
#   bash scripts/fresh_init_publish.sh --no-push   # rebuild only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE_NAME="${REMOTE_NAME:-vislanding}"
REMOTE_URL="${REMOTE_URL:-git@github.com:Victoire7/Vislanding.git}"
NO_PUSH=0
for arg in "$@"; do
  [[ "$arg" == "--no-push" ]] && NO_PUSH=1
done

GIT=/usr/bin/git

CORE_PATHS=(
  .gitignore
  LICENSE
  README.md
  requirements.txt
  hubconf.py
  safe_pre_demo.py
  safe_pre_mask.py
  eval_safe.py
  eval_safe_multi.py
  docs/
  scripts/publish_vislanding.sh
  scripts/prune_for_main.sh
  scripts/fresh_init_publish.sh
  mono/
  deployment/
  media/screenshots/
  weight/README.md
  final_model/README.md
  training/README.md
  training/__init__.py
  training/kitti_json_files/
  training/data_server_info/
  training/mono/
)

TEST_EXTRA_PATHS=(
  finetune.sh
  hubconf2.py
  cal_area_demo.py
  cal_area_demo_real.py
  vis_gt.py
  vis_3d_bar.py
  vis_miou_bar.py
  make_video.py
  data/
  onnx/
  media/gifs/
  iros_exp/
  eval_semantic/
  training/scripts/
  training/mono/configs/ablation/
  training/mono/configs/semi/
  training/mono/configs/0226/
  training/mono/configs/2026_new/
)

stage_core() {
  for p in "${CORE_PATHS[@]}"; do
    if [[ -e "$p" ]]; then
      $GIT add "$p"
    else
      echo "  [skip] $p"
    fi
  done
}

stage_test_extras() {
  for p in "${TEST_EXTRA_PATHS[@]}"; do
    if [[ -e "$p" ]]; then
      $GIT add -f "$p" 2>/dev/null || $GIT add "$p"
    else
      echo "  [skip] $p"
    fi
  done
}

strip_artifacts() {
  $GIT rm -r --cached weight/*.pth final_model/*.pth training/work_dirs 2>/dev/null || true
  $GIT rm -rf --cached training/scripts/depth_to_normal 2>/dev/null || true
  $GIT rm -r --cached .idea training/scripts/result 2>/dev/null || true
  $GIT rm -f --cached docs/METRIC3D_README.md 2>/dev/null || true
}

echo "[fresh] remote=$REMOTE_NAME"
if $GIT remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  $GIT remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  $GIT remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

echo "[fresh] orphan branch: main"
$GIT checkout --orphan _fresh_main
$GIT reset
stage_core
bash scripts/prune_for_main.sh
strip_artifacts
$GIT commit -m "Initial commit"
$GIT branch -D main 2>/dev/null || true
$GIT branch -m main

echo "[fresh] orphan branch: test"
$GIT checkout --orphan _fresh_test
$GIT reset
stage_core
stage_test_extras
strip_artifacts
$GIT commit -m "Initial commit"
$GIT branch -D test 2>/dev/null || true
$GIT branch -m test

$GIT checkout main

for b in release/vislanding onnx_support; do
  $GIT branch -D "$b" 2>/dev/null || true
done

echo "[fresh] prune local history objects..."
$GIT reflog expire --expire=now --all
$GIT gc --prune=now --aggressive

echo "[fresh] main: $($GIT rev-list --count main) commit(s)"
echo "[fresh] test: $($GIT rev-list --count test) commit(s)"

[[ "$NO_PUSH" -eq 1 ]] && { echo "[fresh] --no-push done."; exit 0; }

echo "[fresh] delete stale remote branches..."
for rb in release/vislanding onnx_support; do
  $GIT push "$REMOTE_NAME" --delete "$rb" 2>/dev/null || true
done

echo "[fresh] push main + test (force)..."
$GIT push -u "$REMOTE_NAME" main --force
$GIT push -u "$REMOTE_NAME" test --force

echo "[fresh] done: $REMOTE_URL"
echo "  main -> $($GIT rev-parse --short main)"
echo "  test -> $($GIT rev-parse --short test)"
