#!/usr/bin/env python3
"""
Numerical sanity check: PyTorch eval forward vs ONNX (FP32 / FP16) forward.

Runs the same input through both and prints per-output max-abs / mean-abs
diff. Useful right after exporting to confirm the patches did not silently
change the network's behaviour.

Usage:
    python deployment/jetson/verify_onnx.py \\
        --cfg  mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py \\
        --ckpt /path/to/best.pth \\
        --onnx safe_metric3d_vit_small.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from mmengine import Config
except ImportError:
    from mmcv import Config  # type: ignore

from mono.model.monodepth_model import get_configured_monodepth_model

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from safe_metric3d_export import (  # noqa: E402
    SafeMetric3DOnnxWrapper,
    _export_patches,
    _patch_decoder_get_bins,
    _patch_vit_interpolate_pos_encoding,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--height", type=int, default=616)
    p.add_argument("--width", type=int, default=1064)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = Config.fromfile(args.cfg)
    model = get_configured_monodepth_model(cfg)
    blob = torch.load(args.ckpt, map_location="cpu")
    sd = blob.get("model_state_dict", blob.get("state_dict", blob))
    model.load_state_dict(sd, strict=False)
    model.eval()

    dense = model.depth_model
    _patch_vit_interpolate_pos_encoding(dense)
    _patch_decoder_get_bins(dense.decoder, device)
    wrapper = SafeMetric3DOnnxWrapper(dense).to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    image = rng.uniform(0, 255, size=(1, 3, args.height, args.width)).astype(np.float32)
    image_t = torch.from_numpy(image).to(device)

    with _export_patches(verbose=False):
        # Pre-warm to ensure the dynamic depth_expectation_anchor is registered
        with torch.no_grad():
            _ = wrapper(image_t)

        with torch.no_grad():
            depth_pt, conf_pt, normal_pt, safe_logits_pt, safe_prob_pt = wrapper(image_t)

    pt_outs = {
        "pred_depth": depth_pt.cpu().float().numpy(),
        "confidence": conf_pt.cpu().float().numpy(),
        "pred_normal": normal_pt.cpu().float().numpy(),
        "safe_logits": safe_logits_pt.cpu().float().numpy(),
        "safe_prob": safe_prob_pt.cpu().float().numpy(),
    }

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (
        [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=providers)
    in_meta = sess.get_inputs()[0]
    onnx_input = image.astype(np.float16) if "float16" in in_meta.type else image.astype(np.float32)
    onnx_outs_raw = sess.run(None, {in_meta.name: onnx_input})
    out_names = [o.name for o in sess.get_outputs()]
    onnx_outs = {n: np.asarray(v, dtype=np.float32) for n, v in zip(out_names, onnx_outs_raw)}

    print("=" * 72)
    print(f"{'output':14s}  {'shape':22s}  {'max|diff|':>11s}  {'mean|diff|':>11s}  {'rel%':>7s}")
    print("-" * 72)
    for name, pt in pt_outs.items():
        if name not in onnx_outs:
            continue
        on = onnx_outs[name]
        if on.shape != pt.shape:
            print(f"{name:14s}  shape mismatch: pt={pt.shape} onnx={on.shape}")
            continue
        diff = np.abs(on - pt)
        denom = np.abs(pt).mean() + 1e-8
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        rel = mean_d / denom * 100
        print(f"{name:14s}  {str(on.shape):22s}  {max_d:11.5f}  {mean_d:11.5f}  {rel:7.3f}%")
    print("=" * 72)
    print("ok if max|diff| < ~1e-3 (FP32) or < ~5e-2 (FP16)")


if __name__ == "__main__":
    main()
