#!/usr/bin/env python3
"""
ONNXRuntime inference for Safe Metric3D on Jetson (or any CUDA host).

Supports three providers:
  * --provider cuda      CUDAExecutionProvider (vanilla)
  * --provider tensorrt  TensorrtExecutionProvider with FP16 (Jetson default)
  * --provider cpu       CPUExecutionProvider (sanity check)

The exported graph already contains the T RAFT iterations unrolled, so a single
`session.run` covers all T steps; no Python loop here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

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


def _build_session(onnx_path: str, provider: str, workspace_dir: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "EXHAUSTIVE"}),
            "CPUExecutionProvider",
        ]
    elif provider == "tensorrt":
        cache_dir = os.path.join(workspace_dir, "trt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_max_workspace_size": str(2 * 1024 * 1024 * 1024),
                },
            ),
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"unknown provider: {provider}")

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    print(f"[ort] providers in use: {sess.get_providers()}")
    for inp in sess.get_inputs():
        print(f"[ort] input  {inp.name}  dtype={inp.type}  shape={inp.shape}")
    for out in sess.get_outputs():
        print(f"[ort] output {out.name}  dtype={out.type}  shape={out.shape}")
    return sess


def _to_io_dtype(arr: np.ndarray, sess_input) -> np.ndarray:
    """ORT may have float16 IO if --full-fp16 was used at conversion time."""
    if "float16" in sess_input.type:
        return arr.astype(np.float16, copy=False)
    return arr.astype(np.float32, copy=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None, help="output dir (default: alongside the image)")
    p.add_argument("--height", type=int, default=INPUT_SIZE_VIT[0])
    p.add_argument("--width", type=int, default=INPUT_SIZE_VIT[1])
    p.add_argument(
        "--intrinsic",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        default=None,
        help="optional camera intrinsics [fx fy cx cy] in pixels (original image)",
    )
    p.add_argument("--provider", choices=["cuda", "tensorrt", "cpu"], default="tensorrt")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=1, help="repeat the inference N times for timing")
    p.add_argument("--threshold", type=float, default=0.5, help="safe-prob threshold for binary mask")
    return p.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"failed to read image: {args.image}")

    intrinsic = list(args.intrinsic) if args.intrinsic else None
    x, meta = preprocess(img, intrinsic, is_bgr=True, canonical_hw=(args.height, args.width))

    workspace_dir = os.path.dirname(os.path.abspath(args.onnx))
    sess = _build_session(args.onnx, args.provider, workspace_dir)
    in_meta = sess.get_inputs()[0]
    feed = {in_meta.name: _to_io_dtype(x, in_meta)}

    for _ in range(max(0, args.warmup)):
        sess.run(None, feed)

    start = time.time()
    for _ in range(args.repeat):
        outs = sess.run(None, feed)
    dt = (time.time() - start) / max(1, args.repeat)
    print(f"[time] mean inference: {dt * 1000:.1f} ms over {args.repeat} runs (provider={args.provider})")

    out_names: List[str] = [o.name for o in sess.get_outputs()]
    out_dict = dict(zip(out_names, outs))
    pred_depth = out_dict["pred_depth"]
    confidence = out_dict["confidence"]
    pred_normal = out_dict["pred_normal"]
    safe_logits = out_dict["safe_logits"]

    depth_metric = postprocess_depth(pred_depth.astype(np.float32), meta, use_metric=intrinsic is not None)
    safe = postprocess_safe(safe_logits.astype(np.float32), meta, threshold=args.threshold)
    normal = postprocess_normal(pred_normal.astype(np.float32), meta)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.image)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    np.savez_compressed(
        os.path.join(out_dir, f"{stem}_predictions.npz"),
        depth=depth_metric,
        confidence=confidence[0, 0].astype(np.float32),
        normal=normal["normal"],
        kappa=normal["kappa"],
        safe_prob=safe["prob"],
        safe_mask=safe["mask"],
    )

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
