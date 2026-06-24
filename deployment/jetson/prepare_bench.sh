#!/usr/bin/env bash
#
# === Host side (CUDA dev machine) ===
# Export FP32 ONNX for each T in ITERS_LIST. The same FP32 ONNX is later used
# by Jetson's bench_fps.py for both FP32 and FP16 engines (trtexec handles the
# precision internally; no need to pre-convert the ONNX to FP16).
#
# Usage:
#   ./prepare_bench.sh [output_dir]
#
# Override defaults inline:
#   CKPT=/path/x.pth ITERS_LIST="1 2 4 8" ./prepare_bench.sh ./bench_models

set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/defaults.env"

OUTPUT_DIR="${1:-$REPO_ROOT/safe_metric3d_bench_models}"
mkdir -p "$OUTPUT_DIR"

echo "[prepare_bench] cfg     : $CFG"
echo "[prepare_bench] ckpt    : $CKPT"
echo "[prepare_bench] input   : ${HEIGHT}x${WIDTH}"
echo "[prepare_bench] T values: $ITERS_LIST"
echo "[prepare_bench] output  : $OUTPUT_DIR"

if [[ ! -f "$REPO_ROOT/$CFG" && ! -f "$CFG" ]]; then
  echo "ERROR: cfg not found: $CFG" >&2; exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: ckpt not found: $CKPT" >&2; exit 1
fi

cd "$REPO_ROOT"

for T in $ITERS_LIST; do
  out="$OUTPUT_DIR/safe_metric3d_T${T}.onnx"
  if [[ -f "$out" ]]; then
    echo "[prepare_bench] skip (exists): $out"
    continue
  fi
  echo "[prepare_bench] exporting T=${T} -> $out"
  python deployment/jetson/safe_metric3d_export.py \
    --cfg    "$CFG" \
    --ckpt   "$CKPT" \
    --output "$out" \
    --height "$HEIGHT" \
    --width  "$WIDTH" \
    --iters  "$T" \
    --opset 14
done

echo "[prepare_bench] all ONNX files in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/safe_metric3d_T*.onnx
