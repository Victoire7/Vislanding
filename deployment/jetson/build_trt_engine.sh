#!/usr/bin/env bash
#
# Build a TensorRT engine from the Safe Metric3D ONNX.
#
# Usage:
#   build_trt_engine.sh model.onnx model.plan [extra trtexec args ...]
#
# Examples:
#   build_trt_engine.sh model.onnx model.plan --fp16
#   build_trt_engine.sh model.onnx model.plan --fp16 --useCudaGraph
#   build_trt_engine.sh model.onnx model.plan --int8 --calib=calib.cache
#   build_trt_engine.sh model.onnx model.plan --useDLACore=0 --allowGPUFallback
#
# Defaults: --fp16, builder optimization level 5, ~4 GB workspace, CUDA graph enabled.
# Pass --no-cuda-graph (consumed by this wrapper, not forwarded) to disable it.
set -euo pipefail

ONNX="${1:?usage: $0 model.onnx output.plan [extra trtexec args]}"
OUT="${2:-${ONNX%.onnx}.plan}"
shift 2 || true

if [[ ! -f "$ONNX" ]]; then
  echo "ERROR: ONNX not found: $ONNX" >&2
  exit 1
fi

TRTEXEC=""
for c in /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt/bin/trtexec trtexec; do
  if [[ -x "$c" ]]; then
    TRTEXEC="$c"
    break
  fi
  if command -v "$c" >/dev/null 2>&1; then
    TRTEXEC="$c"
    break
  fi
done

if [[ -z "$TRTEXEC" ]]; then
  cat >&2 <<'EOF'
trtexec not found. Common locations:
  - Jetson:  /usr/src/tensorrt/bin/trtexec
  - x86:     ${TRT_ROOT}/bin/trtexec
Set PATH or symlink trtexec into your PATH and retry.
EOF
  exit 1
fi

# Detect whether trtexec uses --memPoolSize (TRT >=8.4) or --workspace (older).
TRT_VERSION_FULL="$("$TRTEXEC" --version 2>&1 | head -n 1 || true)"
TRT_HAS_MEMPOOL="false"
if "$TRTEXEC" --help 2>&1 | grep -qE -- "--memPoolSize"; then
  TRT_HAS_MEMPOOL="true"
fi

# --- defaults ---
USE_CUDA_GRAPH="true"
USER_ARGS=()
HAS_FP_FLAG="false"
for arg in "$@"; do
  case "$arg" in
    --no-cuda-graph)
      USE_CUDA_GRAPH="false"
      ;;
    --fp16|--int8|--best|--bf16|--noTF32)
      HAS_FP_FLAG="true"
      USER_ARGS+=("$arg")
      ;;
    *)
      USER_ARGS+=("$arg")
      ;;
  esac
done

CMD=( "$TRTEXEC" --onnx="$ONNX" --saveEngine="$OUT" --verbose --buildOnly )

# Default to FP16 unless caller already requested a precision flag.
if [[ "$HAS_FP_FLAG" == "false" ]]; then
  CMD+=( --fp16 )
fi

if [[ "$TRT_HAS_MEMPOOL" == "true" ]]; then
  CMD+=( --memPoolSize=workspace:4096 )
else
  CMD+=( --workspace=4096 )
fi

# Builder optimization level (TRT 9+; ignored by older trtexec which will warn).
if "$TRTEXEC" --help 2>&1 | grep -qE -- "--builderOptimizationLevel"; then
  CMD+=( --builderOptimizationLevel=5 )
fi

if [[ "$USE_CUDA_GRAPH" == "true" ]] && "$TRTEXEC" --help 2>&1 | grep -qE -- "--useCudaGraph"; then
  CMD+=( --useCudaGraph )
fi

CMD+=( "${USER_ARGS[@]}" )

echo "[build_trt_engine] trtexec: $TRTEXEC"
echo "[build_trt_engine] $(printf ' %q' "${CMD[@]}")"
"${CMD[@]}"

echo "[build_trt_engine] engine saved to $OUT"
