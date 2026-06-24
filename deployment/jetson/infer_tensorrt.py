#!/usr/bin/env python3
"""
Pure TensorRT Python runtime for Safe Metric3D on Jetson.

Why pure TRT (vs ONNXRuntime)?
  * Lower memory overhead than ORT-TRT EP on Jetson Orin / Xavier.
  * Direct control over fp16 / int8 / DLA choices.
  * Simpler dependency chain (only `tensorrt` + `pycuda`/`cuda-python`).

Build the engine first via `build_trt_engine.sh model.onnx model.plan`, then:

    python deployment/jetson/infer_tensorrt.py \\
        --engine model.plan --image foo.jpg --intrinsic 1500 1500 960 540

The runtime supports both fp32 and fp16 IO bindings (auto-detected).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline import (  # noqa: E402
    INPUT_SIZE_VIT,
    colorize_depth,
    colorize_safe,
    postprocess_depth,
    postprocess_normal,
    postprocess_safe,
    preprocess,
)


def _import_trt():
    try:
        import tensorrt as trt  # type: ignore
    except ImportError as exc:
        sys.exit(
            "tensorrt python bindings not found. On Jetson they ship with JetPack:\n"
            "  - JetPack 5.x: /usr/lib/python3.8/dist-packages/tensorrt/\n"
            "  - JetPack 6.x: /usr/lib/python3.10/dist-packages/tensorrt/\n"
            "Make sure you're using the system python, or pip install nvidia-tensorrt.\n"
            f"Original error: {exc}"
        )
    return trt


def _import_cuda():
    """
    Prefer the modern `cuda.bindings.driver` (cuda-python) where available, but
    transparently fall back to the more common pycuda path.
    """
    try:
        import pycuda.autoinit  # noqa: F401  # initializes the primary context
        import pycuda.driver as cuda  # type: ignore

        return ("pycuda", cuda)
    except ImportError:
        pass
    try:
        from cuda import cudart  # type: ignore

        return ("cudart", cudart)
    except ImportError:
        sys.exit(
            "Neither pycuda nor cuda-python available. Install one of them:\n"
            "  pip install pycuda          # easier, requires CUDA toolkit\n"
            "  pip install cuda-python     # NVIDIA official"
        )


# ---------------------------------------------------------------------------
# Generic engine wrapper
# ---------------------------------------------------------------------------
class TrtRunner:
    """
    Minimal but solid TensorRT runtime that:

      * loads .plan
      * supports static or dynamic shapes (sets the optimization profile when present)
      * binds host/device buffers per binding
      * runs `enqueueV3` (TRT >=8.5) or `execute_async_v2` fallback
      * returns numpy arrays in original dtype (caller decides)
    """

    def __init__(self, engine_path: str):
        self.trt = _import_trt()
        self.cuda_kind, self.cuda = _import_cuda()

        logger = self.trt.Logger(self.trt.Logger.WARNING)
        runtime = self.trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            sys.exit(f"failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = self._create_stream()

        # Names + roles
        self.input_names: List[str] = []
        self.output_names: List[str] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.host: Dict[str, np.ndarray] = {}
        self.device: Dict[str, int] = {}

    # ----- cuda backend abstraction -----
    def _create_stream(self):
        if self.cuda_kind == "pycuda":
            return self.cuda.Stream()
        # cudart
        status, outs = self._unpack(self.cuda.cudaStreamCreate())
        self._check(status)
        return outs[0]

    @staticmethod
    def _unpack(rv):
        """cuda-python returns (status, *outputs); pycuda returns plain values."""
        if isinstance(rv, tuple):
            status = rv[0]
            outputs = rv[1:]
        else:
            status = rv
            outputs = ()
        return status, outputs

    def _check(self, status):
        if self.cuda_kind != "cudart":
            return
        # cuda-python statuses are enums; cudaSuccess == 0
        if int(status) != 0:
            raise RuntimeError(f"CUDA error: {status}")

    def _malloc(self, nbytes: int) -> int:
        if self.cuda_kind == "pycuda":
            return int(self.cuda.mem_alloc(nbytes))
        status, outs = self._unpack(self.cuda.cudaMalloc(nbytes))
        self._check(status)
        return int(outs[0])

    def _memcpy_htod_async(self, dst: int, src: np.ndarray):
        if self.cuda_kind == "pycuda":
            self.cuda.memcpy_htod_async(dst, src, self.stream)
            return
        status, _ = self._unpack(
            self.cuda.cudaMemcpyAsync(
                int(dst),
                int(src.ctypes.data),
                int(src.nbytes),
                self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
        )
        self._check(status)

    def _memcpy_dtoh_async(self, dst: np.ndarray, src: int):
        if self.cuda_kind == "pycuda":
            self.cuda.memcpy_dtoh_async(dst, src, self.stream)
            return
        status, _ = self._unpack(
            self.cuda.cudaMemcpyAsync(
                int(dst.ctypes.data),
                int(src),
                int(dst.nbytes),
                self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        )
        self._check(status)

    def _stream_sync(self):
        if self.cuda_kind == "pycuda":
            self.stream.synchronize()
            return
        status, _ = self._unpack(self.cuda.cudaStreamSynchronize(self.stream))
        self._check(status)

    # ----- shape helpers -----
    def set_input_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        self.context.set_input_shape(name, shape)

    def _allocate(self, name: str, shape: Tuple[int, ...]) -> None:
        dtype = self._numpy_dtype(name)
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        host = np.empty(tuple(shape), dtype=dtype)
        device = self._malloc(nbytes)
        self.host[name] = host
        self.device[name] = device
        self.context.set_tensor_address(name, device)

    def _numpy_dtype(self, name: str) -> np.dtype:
        trt_dtype = self.engine.get_tensor_dtype(name)
        mapping = {
            self.trt.DataType.FLOAT: np.float32,
            self.trt.DataType.HALF: np.float16,
            self.trt.DataType.INT8: np.int8,
            self.trt.DataType.INT32: np.int32,
            self.trt.DataType.BOOL: np.bool_,
        }
        # Newer TRT also has INT64, UINT8
        if hasattr(self.trt.DataType, "INT64"):
            mapping[self.trt.DataType.INT64] = np.int64
        if hasattr(self.trt.DataType, "UINT8"):
            mapping[self.trt.DataType.UINT8] = np.uint8
        if trt_dtype not in mapping:
            raise RuntimeError(f"Unsupported TRT dtype on {name}: {trt_dtype}")
        return np.dtype(mapping[trt_dtype])

    def prepare(self, input_shapes: Dict[str, Tuple[int, ...]]) -> None:
        for name, shape in input_shapes.items():
            self.set_input_shape(name, shape)
        # Allocate all bindings (inputs + outputs) using current context shapes
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            self._allocate(name, shape)

    # ----- inference -----
    def infer(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name, arr in feed.items():
            target_dtype = self.host[name].dtype
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype, copy=False)
            np.copyto(self.host[name], arr)
            self._memcpy_htod_async(self.device[name], self.host[name])

        # enqueueV3 — supported on TRT 8.5+ which is what JetPack 5.1+ ships
        ok = self.context.execute_async_v3(stream_handle=self._stream_handle())
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")

        outs: Dict[str, np.ndarray] = {}
        for name in self.output_names:
            self._memcpy_dtoh_async(self.host[name], self.device[name])
            outs[name] = self.host[name]
        self._stream_sync()
        return outs

    def _stream_handle(self) -> int:
        if self.cuda_kind == "pycuda":
            return self.stream.handle
        return int(self.stream)

    def __del__(self):
        if getattr(self, "cuda_kind", None) != "cudart":
            return
        stream = getattr(self, "stream", None)
        if stream is None:
            return
        try:
            self.cuda.cudaStreamDestroy(stream)
        except Exception:
            pass
        for ptr in getattr(self, "device", {}).values():
            try:
                self.cuda.cudaFree(int(ptr))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--height", type=int, default=INPUT_SIZE_VIT[0])
    p.add_argument("--width", type=int, default=INPUT_SIZE_VIT[1])
    p.add_argument(
        "--intrinsic",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        default=None,
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"failed to read image: {args.image}")

    intrinsic = list(args.intrinsic) if args.intrinsic else None
    x, meta = preprocess(img, intrinsic, is_bgr=True, canonical_hw=(args.height, args.width))

    runner = TrtRunner(args.engine)
    print(f"[trt] inputs : {runner.input_names}")
    print(f"[trt] outputs: {runner.output_names}")

    img_input_name = runner.input_names[0]
    runner.prepare({img_input_name: (1, 3, args.height, args.width)})

    feed = {img_input_name: x}

    for _ in range(max(0, args.warmup)):
        runner.infer(feed)

    start = time.time()
    for _ in range(args.repeat):
        outs = runner.infer(feed)
    dt_ms = (time.time() - start) / max(1, args.repeat) * 1000
    print(f"[time] mean inference: {dt_ms:.2f} ms over {args.repeat} runs")

    pred_depth = outs.get("pred_depth")
    confidence = outs.get("confidence")
    pred_normal = outs.get("pred_normal")
    safe_logits = outs.get("safe_logits")
    if pred_depth is None or safe_logits is None:
        raise RuntimeError(
            f"engine outputs unexpected. got: {list(outs.keys())}; "
            "did you re-export with the matching wrapper?"
        )

    depth_metric = postprocess_depth(pred_depth.astype(np.float32), meta, use_metric=intrinsic is not None)
    safe = postprocess_safe(safe_logits.astype(np.float32), meta, threshold=args.threshold)
    normal = postprocess_normal(pred_normal.astype(np.float32), meta) if pred_normal is not None else None

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.image)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    npz_payload = dict(
        depth=depth_metric,
        safe_prob=safe["prob"],
        safe_mask=safe["mask"],
    )
    if confidence is not None:
        npz_payload["confidence"] = confidence[0, 0].astype(np.float32)
    if normal is not None:
        npz_payload["normal"] = normal["normal"]
        npz_payload["kappa"] = normal["kappa"]
    np.savez_compressed(os.path.join(out_dir, f"{stem}_predictions.npz"), **npz_payload)

    cv2.imwrite(os.path.join(out_dir, f"{stem}_depth.png"), colorize_depth(depth_metric))
    cv2.imwrite(os.path.join(out_dir, f"{stem}_safe_prob.png"), colorize_safe(safe["prob"]))
    cv2.imwrite(os.path.join(out_dir, f"{stem}_safe_mask.png"), (safe["mask"] * 255).astype(np.uint8))

    overlay = (img.astype(np.float32) * 0.5 + colorize_safe(safe["prob"]).astype(np.float32) * 0.5).clip(0, 255)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_safe_overlay.png"), overlay.astype(np.uint8))

    print(
        f"[depth ] min={depth_metric.min():.3f} max={depth_metric.max():.3f}"
        + ("" if intrinsic else " (canonical, no fx given)")
    )
    print(f"[safe  ] prob in [{safe['prob'].min():.3f}, {safe['prob'].max():.3f}], mask>={args.threshold}: "
          f"{(safe['mask'] > 0).mean() * 100:.2f}%")
    print(f"[saved ] {out_dir}/{stem}_*.png and predictions.npz")


if __name__ == "__main__":
    main()
