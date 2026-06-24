"""
Common pre/post-processing utilities for Safe Metric3D Jetson deployment.

The pipeline mirrors `eval_safe.py` so that ONNX / TensorRT outputs match the
PyTorch eval path numerically (modulo FP16 noise):

  preprocess(image_bgr, intrinsic) ->
      input_tensor [1,3,H,W] (RGB, raw [0,255], padded with ImageNet mean)
      meta = (pad_info, scale_factor, scaled_intrinsic, original_hw)

  postprocess_depth(pred_depth_padded, meta) -> metric depth at original size
  postprocess_safe(safe_logits_padded, meta)  -> {logits, prob, mask} at original size
  postprocess_normal(pred_normal_padded, meta) -> normal map (3ch) + kappa (1ch) at original size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

INPUT_SIZE_VIT = (616, 1064)  # (H, W) -- canonical Metric3D ViT-small input
RGB_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RGB_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
CANONICAL_FOCAL = 1000.0  # mono.configs ... canonical_space.focal_length


@dataclass
class FrameMeta:
    pad_info: List[int]                 # [top, bottom, left, right]
    scale: float                        # scale = min(H_canon/h, W_canon/w) applied before padding
    intrinsic_scaled: Optional[List[float]]  # [fx', fy', cx', cy'] after scale
    original_hw: Tuple[int, int]
    canonical_hw: Tuple[int, int]


def preprocess(
    image_bgr_or_rgb: np.ndarray,
    intrinsic: Optional[List[float]] = None,
    *,
    is_bgr: bool = True,
    canonical_hw: Tuple[int, int] = INPUT_SIZE_VIT,
) -> Tuple[np.ndarray, FrameMeta]:
    """
    Resize-keep-aspect to fit canonical_hw, then pad with the ImageNet mean (raw RGB,
    NOT yet normalized) and pack into NCHW float32. The exported ONNX model is wrapped
    with a normalization layer, so the network receives raw [0,255] RGB.

    Args:
        image_bgr_or_rgb: HxWx3 uint8 image
        intrinsic: optional [fx, fy, cx, cy] in pixels (original image)
        is_bgr: True if input is BGR (e.g. cv2.imread). Will be flipped to RGB.

    Returns:
        x: float32 NCHW [1,3,H,W], dtype float32
        meta: pad_info / scale / intrinsic_scaled / original_hw / canonical_hw
    """
    if image_bgr_or_rgb.dtype != np.uint8:
        image_bgr_or_rgb = image_bgr_or_rgb.astype(np.uint8)

    rgb = image_bgr_or_rgb[:, :, ::-1] if is_bgr else image_bgr_or_rgb
    rgb = np.ascontiguousarray(rgb)

    h0, w0 = rgb.shape[:2]
    th, tw = canonical_hw
    scale = float(min(th / h0, tw / w0))

    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = th - new_h
    pad_w = tw - new_w
    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_b = pad_h - pad_t
    pad_r = pad_w - pad_l

    rgb = cv2.copyMakeBorder(
        rgb,
        pad_t, pad_b, pad_l, pad_r,
        cv2.BORDER_CONSTANT,
        value=RGB_MEAN.tolist(),
    )

    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32, copy=False))

    intr_scaled = None
    if intrinsic is not None:
        assert len(intrinsic) == 4, "intrinsic must be [fx, fy, cx, cy]"
        intr_scaled = [float(v) * scale for v in intrinsic]

    meta = FrameMeta(
        pad_info=[pad_t, pad_b, pad_l, pad_r],
        scale=scale,
        intrinsic_scaled=intr_scaled,
        original_hw=(h0, w0),
        canonical_hw=(th, tw),
    )
    return x, meta


def _crop_pad_2d(arr: np.ndarray, meta: FrameMeta) -> np.ndarray:
    """Strip the padding from a CHW-layout (or HW) tensor sized canonical_hw."""
    pt, pb, pl, pr = meta.pad_info
    th, tw = meta.canonical_hw
    h_slice = slice(pt, th - pb)
    w_slice = slice(pl, tw - pr)
    if arr.ndim == 2:
        return arr[h_slice, w_slice]
    if arr.ndim == 3:
        return arr[:, h_slice, w_slice]
    if arr.ndim == 4:
        return arr[:, :, h_slice, w_slice]
    raise ValueError(f"unexpected rank: {arr.shape}")


def _resize_to_original_2d(arr2d: np.ndarray, meta: FrameMeta, *, mode: int = cv2.INTER_LINEAR) -> np.ndarray:
    oh, ow = meta.original_hw
    return cv2.resize(arr2d, (ow, oh), interpolation=mode)


def postprocess_depth(
    pred_depth: np.ndarray,
    meta: FrameMeta,
    *,
    use_metric: bool = True,
) -> np.ndarray:
    """
    pred_depth: [1,1,H,W] or [H,W] in canonical space (output of decoder.prediction).
    Returns 2D float32 array sized to the original input image.

    If `use_metric=True` and `meta.intrinsic_scaled` is provided, scale to metric depth
    using fx_scaled / 1000 (the canonical focal length).
    """
    d = np.asarray(pred_depth)
    if d.ndim == 4:
        d = d[0, 0]
    elif d.ndim == 3:
        d = d[0]
    elif d.ndim != 2:
        raise ValueError(f"unexpected depth shape: {d.shape}")

    d = _crop_pad_2d(d, meta)
    d = _resize_to_original_2d(d, meta, mode=cv2.INTER_LINEAR)

    if use_metric and meta.intrinsic_scaled is not None:
        fx_scaled = meta.intrinsic_scaled[0]
        d = d.astype(np.float32) * (fx_scaled / CANONICAL_FOCAL)

    return d.astype(np.float32, copy=False)


def postprocess_safe(
    safe_logits: np.ndarray,
    meta: FrameMeta,
    *,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    safe_logits: [1,2,H,W] (channel 0 = unsafe, channel 1 = safe by training convention).
    Returns dict with keys: logits (2,H,W), prob (H,W), mask (H,W) in original size.
    """
    sl = np.asarray(safe_logits)
    if sl.ndim != 4 or sl.shape[1] != 2:
        raise ValueError(f"safe_logits expected [1,2,H,W], got {sl.shape}")
    sl = sl[0]

    sl = _crop_pad_2d(sl, meta)

    diff = sl[1] - sl[0]
    prob = 1.0 / (1.0 + np.exp(-diff.astype(np.float32)))

    logits_orig = np.stack(
        [
            _resize_to_original_2d(sl[0].astype(np.float32), meta, mode=cv2.INTER_LINEAR),
            _resize_to_original_2d(sl[1].astype(np.float32), meta, mode=cv2.INTER_LINEAR),
        ],
        axis=0,
    )
    prob_orig = _resize_to_original_2d(prob, meta, mode=cv2.INTER_LINEAR)
    mask_orig = (prob_orig >= threshold).astype(np.uint8)

    return {"logits": logits_orig, "prob": prob_orig, "mask": mask_orig}


def postprocess_normal(pred_normal: np.ndarray, meta: FrameMeta) -> Dict[str, np.ndarray]:
    """pred_normal: [1,4,H,W] (nx, ny, nz, kappa) → return normal (3,H,W) + kappa (H,W) at original size."""
    n = np.asarray(pred_normal)
    if n.ndim != 4 or n.shape[1] != 4:
        raise ValueError(f"pred_normal expected [1,4,H,W], got {n.shape}")
    n = n[0]
    n = _crop_pad_2d(n, meta)
    normal_xyz = np.stack(
        [_resize_to_original_2d(n[i].astype(np.float32), meta, mode=cv2.INTER_LINEAR) for i in range(3)],
        axis=0,
    )
    norm = np.linalg.norm(normal_xyz, axis=0, keepdims=True) + 1e-10
    normal_xyz = normal_xyz / norm
    kappa = _resize_to_original_2d(n[3].astype(np.float32), meta, mode=cv2.INTER_LINEAR)
    return {"normal": normal_xyz.astype(np.float32), "kappa": kappa.astype(np.float32)}


def colorize_depth(depth: np.ndarray, *, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Returns a uint8 BGR colormap visualization."""
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if vmin is None:
        vmin = float(np.percentile(d[valid], 2)) if valid.any() else 0.0
    if vmax is None:
        vmax = float(np.percentile(d[valid], 98)) if valid.any() else 1.0
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def colorize_safe(prob: np.ndarray) -> np.ndarray:
    """probability of `safe` class -> heatmap BGR."""
    p = np.clip(prob, 0.0, 1.0).astype(np.float32)
    return cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
