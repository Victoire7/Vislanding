#!/usr/bin/env bash
# Publish Vislanding to GitHub (main = core, test = core + auxiliary).
#
# Usage:
#   bash scripts/publish_vislanding.sh main
#   bash scripts/publish_vislanding.sh test
#   bash scripts/publish_vislanding.sh main --no-push
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE_NAME="${REMOTE_NAME:-vislanding}"
REMOTE_URL="${REMOTE_URL:-git@github.com:Victoire7/Vislanding.git}"

BRANCH="${1:-main}"
shift || true

ADD_ONLY=0
NO_PUSH=0
DELETE_OLD=0
for arg in "$@"; do
  case "$arg" in
    --add-only)   ADD_ONLY=1 ;;
    --no-push)    NO_PUSH=1 ;;
    --delete-old) DELETE_OLD=1 ;;
    main|test)    BRANCH="$arg" ;;
  esac
done

if [[ "$BRANCH" != "main" && "$BRANCH" != "test" ]]; then
  echo "Usage: $0 {main|test} [--add-only] [--no-push] [--delete-old]"
  exit 1
fi

GIT=/usr/bin/git

echo "[publish] branch=$BRANCH  remote=$REMOTE_NAME"

# --- remote ---
if $GIT remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  $GIT remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  $GIT remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

# --- checkout target branch (rebuild from release/vislanding when available) ---
if $GIT show-ref --verify --quiet refs/heads/release/vislanding; then
  $GIT checkout -B "$BRANCH" release/vislanding
elif $GIT show-ref --verify --quiet "refs/heads/$BRANCH"; then
  $GIT checkout "$BRANCH"
else
  $GIT checkout -b "$BRANCH"
fi

$GIT reset HEAD >/dev/null 2>&1 || true

# --- shared core paths ---
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

echo "[publish] git add core paths..."
for p in "${CORE_PATHS[@]}"; do
  [[ -e "$p" ]] && $GIT add "$p" || echo "  [skip] $p"
done

if [[ "$BRANCH" == "main" ]]; then
  echo "[publish] prune non-core paths for main..."
  bash scripts/prune_for_main.sh
elif [[ "$BRANCH" == "test" ]]; then
  echo "[publish] git add test extras..."
  for p in "${TEST_EXTRA_PATHS[@]}"; do
    [[ -e "$p" ]] && $GIT add "$p" || echo "  [skip] $p"
  done
fi

# strip cached large / legacy artifacts
$GIT rm -r --cached weight/*.pth final_model/*.pth training/work_dirs 2>/dev/null || true
$GIT rm -r --cached .idea training/scripts/result training/scripts/depth_to_normal 2>/dev/null || true
$GIT rm -f --cached docs/METRIC3D_README.md 2>/dev/null || true

echo ""
$GIT diff --cached --stat | tail -25
echo ""
$GIT diff --cached --name-only | grep -E '\.(pth|onnx|plan|pt)$' \
  && { echo "ERROR: weights staged!"; exit 1; } || echo "[ok] no weight files staged"

[[ "$ADD_ONLY" -eq 1 ]] && { echo "[publish] --add-only done."; exit 0; }

if $GIT diff --cached --quiet; then
  echo "[publish] nothing to commit"
else
  $GIT commit -m "Vislanding: update $BRANCH branch"
fi

[[ "$NO_PUSH" -eq 1 ]] && { echo "[publish] --no-push"; exit 0; }

if [[ "$DELETE_OLD" -eq 1 ]] || [[ "$BRANCH" == "main" ]]; then
  echo "[publish] delete remote release/vislanding (if exists)..."
  $GIT push "$REMOTE_NAME" --delete release/vislanding 2>/dev/null || true
fi

echo "[publish] push $BRANCH -> $REMOTE_NAME"
$GIT push -u "$REMOTE_NAME" "$BRANCH" --force
echo "[publish] done: $REMOTE_URL ($BRANCH)"
