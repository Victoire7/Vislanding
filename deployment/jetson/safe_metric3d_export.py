#!/usr/bin/env python3
"""
Export Safe Metric3D (RAFTDepthNormalSafeDPT5 + DINOv2 ViT-small + reg) to ONNX.

This is intentionally robust against the export-time pitfalls in the codebase:

* `xformers.memory_efficient_attention` is disabled (forces standard PyTorch
  attention that's ONNX-traceable).
* `torch.utils.checkpoint.checkpoint` is patched to a passthrough so the trace
  does not contain non-traceable rng-fork machinery.
* `torch.autocast(...)` inside `interpolate_float32` / `upflow4` is bypassed.
* `interpolate_pos_encoding` is rewritten to use bilinear and `antialias=False`
  (TensorRT cannot import the antialias variant).
* Hard-coded `device="cuda"` in `get_bins` and dynamic buffer registration in
  `register_depth_expectation_anchor` are pre-warmed on the export device, so
  the trace produces clean constants.
* All the `torch.isnan(...).any()` / print debug statements inside the decoder
  are silenced during export.

Notes about the RAFT decoder's T-step iteration
-----------------------------------------------
`for itr in range(self.iters)` (T = `cfg.model.decode_head.iters`, default 4 for
the safe-uav config) is unrolled by the TorchScript tracer into T identical
sub-graphs in the ONNX. T is therefore baked into the engine and CAN'T be
changed at runtime. To deploy a different number of iterations re-run this
exporter with a different `--iters` (or edit the config) so the decoder is
rebuilt with the desired T.

Usage
-----
    cd <repo_root>
    python deployment/jetson/safe_metric3d_export.py \\
        --cfg mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py \\
        --ckpt /path/to/your.pth \\
        --output safe_metric3d_vit_small.onnx \\
        --iters 4 \\
        --opset 17

For Jetson + TensorRT it is strongly recommended to keep `--static-shape`
(default) so trtexec can build a fully-shaped fp16 plan without dynamic ranges.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from mmengine import Config
except ImportError:
    from mmcv import Config  # type: ignore

from mono.model.monodepth_model import get_configured_monodepth_model


# ---------------------------------------------------------------------------
# Wrapper that wires (image RGB float32 NCHW raw [0,255]) -> normalized -> net
# Outputs are detached and re-cast to float32 for predictable ONNX dtypes.
# ---------------------------------------------------------------------------
class SafeMetric3DOnnxWrapper(nn.Module):
    output_names = ("pred_depth", "confidence", "pred_normal", "safe_logits", "safe_prob")

    def __init__(self, dense_model: nn.Module):
        super().__init__()
        self.net = dense_model
        self.register_buffer(
            "rgb_mean", torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor):
        x = (image - self.rgb_mean) / self.rgb_std
        out = self.net(x)
        depth = out["prediction"].float()
        conf = out["confidence"].float()
        normal = out["prediction_normal"].float()
        safe_logits = out["safe_prediction"].float()
        safe_prob = torch.sigmoid(safe_logits[:, 1:2] - safe_logits[:, 0:1])
        return depth, conf, normal, safe_logits, safe_prob


# ---------------------------------------------------------------------------
# Export-time patches
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _export_patches(verbose: bool = True) -> Iterator[None]:
    """
    Temporarily patches:
      - xformers' memory_efficient_attention path in the ViT
      - torch.utils.checkpoint.checkpoint -> passthrough
      - torch.autocast inside interpolate_float32 / upflow4 -> direct call
      - decoder isnan/inf debug statements -> no-op
      - ViT.interpolate_pos_encoding -> bilinear, antialias=False

    Yields and restores everything on exit so other parts of the runtime
    (training, etc.) are not affected.
    """
    saved = {}

    # 1) xformers off in the ViT
    try:
        import mono.model.backbones.ViT_DINO_reg as vitmod  # noqa: WPS433
        saved["XFORMERS_AVAILABLE"] = vitmod.XFORMERS_AVAILABLE
        vitmod.XFORMERS_AVAILABLE = False
        if verbose:
            print("[patch] xformers disabled in ViT_DINO_reg")
    except Exception as exc:  # pragma: no cover
        print(f"[patch] failed to disable xformers: {exc}")

    # 2) torch.utils.checkpoint.checkpoint -> passthrough
    import torch.utils.checkpoint as ckpt_mod
    saved["checkpoint"] = ckpt_mod.checkpoint
    def _passthrough(fn, *args, **kwargs):
        kwargs.pop("use_reentrant", None)
        kwargs.pop("preserve_rng_state", None)
        kwargs.pop("context_fn", None)
        kwargs.pop("determinism_check", None)
        kwargs.pop("debug", None)
        return fn(*args, **kwargs)
    ckpt_mod.checkpoint = _passthrough  # type: ignore[assignment]

    # patch the symbols already imported into decoder modules
    decoder_modules = []
    try:
        import mono.model.decode_heads.RAFTDepthNormalSafeDPTDecoder5_bestbak as dec_safe  # noqa: WPS433
        decoder_modules.append(dec_safe)
    except Exception:
        pass
    try:
        import mono.model.decode_heads.RAFTDepthNormalDPTDecoder5_new as dec_new  # noqa: WPS433
        decoder_modules.append(dec_new)
    except Exception:
        pass

    saved["dec_modules_checkpoint"] = []
    for mod in decoder_modules:
        if hasattr(mod, "checkpoint"):
            saved["dec_modules_checkpoint"].append((mod, mod.checkpoint))
            mod.checkpoint = _passthrough  # type: ignore[attr-defined]

    if verbose:
        print(f"[patch] checkpoint -> passthrough across {len(decoder_modules)} decoder modules")

    # 3) bypass torch.autocast within interpolate_float32 / upflow4
    saved["dec_modules_interp"] = []
    for mod in decoder_modules:
        if hasattr(mod, "interpolate_float32"):
            saved["dec_modules_interp"].append((mod, "interpolate_float32", mod.interpolate_float32))

            def _interp(x, size=None, scale_factor=None, mode="nearest", align_corners=None):
                return F.interpolate(
                    x.float(), size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
                )

            mod.interpolate_float32 = _interp  # type: ignore[attr-defined]
        if hasattr(mod, "upflow4"):
            saved["dec_modules_interp"].append((mod, "upflow4", mod.upflow4))

            def _up4(flow, mode="bilinear"):
                new_size = (4 * flow.shape[2], 4 * flow.shape[3])
                return F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

            mod.upflow4 = _up4  # type: ignore[attr-defined]

    # 4) silence the decoder's debug prints + isnan/isinf checks.
    # The decoder calls `if torch.isnan(t).any(): print(...)`.  During ONNX
    # tracing the `if` condition must evaluate to a plain Python bool without
    # involving any trace node; returning torch.tensor(False) still creates a
    # node in the graph and emits TracerWarnings.  Instead we replace isnan /
    # isinf with functions that return a plain Python object whose __bool__ is
    # always False, so the `if` branch is never entered and no trace node is
    # created.  The decoder module's `print` builtin is similarly shadowed.
    class _AlwaysFalseBool:
        """Mimics enough of Tensor so `x.any()` and bool(x) both return False."""
        def any(self):  return self  # noqa: E704
        def all(self):  return self  # noqa: E704
        def __bool__(self): return False  # noqa: E704

    def _noop_isnan(*_a, **_k): return _AlwaysFalseBool()  # noqa: E704
    def _noop_isinf(*_a, **_k): return _AlwaysFalseBool()  # noqa: E704

    saved["dec_modules_print"] = []
    for mod in decoder_modules:
        saved["dec_modules_print"].append((mod, "print" in mod.__dict__, mod.__dict__.get("print")))
        mod.print = lambda *a, **k: None  # type: ignore[attr-defined]

    saved["torch_isnan"] = torch.isnan
    saved["torch_isinf"] = torch.isinf
    torch.isnan = _noop_isnan  # type: ignore[assignment]
    torch.isinf = _noop_isinf  # type: ignore[assignment]

    # 5) patch copy.deepcopy -> tensor.clone() to fix:
    #    "RuntimeError: NYI: Named tensors are not supported with the tracer"
    # The decoder calls `copy.deepcopy(net_list[0])` before the RAFT loop.
    # During tracing the tensor is a TracerTensor; deepcopy on it fails with the
    # above error on PyTorch 1.x/2.x.  For inference, cloning is semantically
    # equivalent and ONNX-traceable.
    import copy as _copy_mod
    _orig_deepcopy = _copy_mod.deepcopy
    saved["deepcopy"] = _orig_deepcopy

    def _tensor_safe_deepcopy(obj, memo=None):
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        return _orig_deepcopy(obj, memo)

    _copy_mod.deepcopy = _tensor_safe_deepcopy  # type: ignore[assignment]
    # Also patch the symbol already imported inside the decoder module(s)
    saved["dec_modules_deepcopy"] = []
    for mod in decoder_modules:
        if hasattr(mod, "copy"):
            saved["dec_modules_deepcopy"].append((mod.copy, mod.copy.deepcopy))
            mod.copy.deepcopy = _tensor_safe_deepcopy  # type: ignore[attr-defined]

    if verbose:
        print("[patch] copy.deepcopy -> tensor.clone() (fixes tracer NYI error)")

    try:
        yield
    finally:
        # restore in reverse order
        _copy_mod.deepcopy = saved.get("deepcopy", _copy_mod.deepcopy)
        for copy_obj, orig_fn in saved.get("dec_modules_deepcopy", []):
            copy_obj.deepcopy = orig_fn
        torch.isnan = saved.get("torch_isnan", torch.isnan)
        torch.isinf = saved.get("torch_isinf", torch.isinf)
        for mod, had_attr, original in saved.get("dec_modules_print", []):
            try:
                if had_attr:
                    setattr(mod, "print", original)
                else:
                    delattr(mod, "print")
            except AttributeError:
                pass
        for mod, name, fn in saved.get("dec_modules_interp", []):
            setattr(mod, name, fn)
        for mod, fn in saved.get("dec_modules_checkpoint", []):
            mod.checkpoint = fn
        ckpt_mod.checkpoint = saved["checkpoint"]
        try:
            import mono.model.backbones.ViT_DINO_reg as vitmod
            vitmod.XFORMERS_AVAILABLE = saved.get("XFORMERS_AVAILABLE", False)
        except Exception:
            pass


def _patch_vit_interpolate_pos_encoding(dense_model: nn.Module) -> None:
    """Bilinear interpolation, antialias=False — required for TensorRT."""
    import math

    encoder = dense_model.encoder

    if not hasattr(encoder, "interpolate_pos_encoding"):
        return

    def interpolate_pos_encoding_static(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0 = w0 + self.interpolate_offset
        h0 = h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    encoder.interpolate_pos_encoding = interpolate_pos_encoding_static.__get__(  # type: ignore[assignment]
        encoder, encoder.__class__
    )
    if hasattr(encoder, "interpolate_antialias"):
        encoder.interpolate_antialias = False


def _patch_decoder_get_bins(decoder: nn.Module, device: torch.device) -> None:
    """Replace the hard-coded `device="cuda"` in get_bins."""
    import math

    def get_bins(self, bins_num):
        depth_bins_vec = torch.linspace(
            math.log(self.min_val), math.log(self.max_val), bins_num, device=device
        )
        return torch.exp(depth_bins_vec)

    decoder.get_bins = get_bins.__get__(decoder, decoder.__class__)  # type: ignore[assignment]


def _prewarm_decoder_buffers(model: nn.Module, dummy_image: torch.Tensor) -> None:
    """Force registration of `depth_expectation_anchor` so the trace sees it as constant."""
    model.eval()
    with torch.no_grad():
        _ = model(dummy_image)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Export Safe Metric3D to ONNX")
    p.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(_REPO_ROOT, "mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py"),
    )
    # Default mirrors safe_pre_demo.py / final_model directory.
    p.add_argument(
        "--ckpt",
        type=str,
        default=os.environ.get(
            "CKPT",
            os.path.join(_REPO_ROOT, "final_model", "student_step00004400_86.04.pth"),
        ),
    )
    p.add_argument("--output", type=str, default="safe_metric3d_vit_small.onnx")
    p.add_argument("--height", type=int, default=616)
    p.add_argument("--width", type=int, default=1064)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--iters", type=int, default=None, help="Override decode_head.iters (default: from cfg)")
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow batch dim to vary at runtime (default: fully static, recommended for TRT)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--strict-state-dict", action="store_true", help="Use strict=True when loading the ckpt")
    p.add_argument("--no-simplify", action="store_true", help="Skip the post-export onnx-simplifier pass")
    return p.parse_args()


def _load_state_dict(model: nn.Module, ckpt_path: str, strict: bool) -> None:
    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict) and "model_state_dict" in blob:
        sd = blob["model_state_dict"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    else:
        sd = blob
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[ckpt] loaded {ckpt_path}")
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU. ONNX export should still succeed,")
        print("       but be advised that the model has CUDA-leaning code paths.")
        args.device = "cpu"
    device = torch.device(args.device)

    cfg = Config.fromfile(args.cfg)
    if args.iters is not None:
        cfg.model.decode_head.iters = int(args.iters)
        print(f"[cfg] override decode_head.iters = {cfg.model.decode_head.iters}")
    model = get_configured_monodepth_model(cfg)
    _load_state_dict(model, args.ckpt, strict=args.strict_state_dict)
    model.eval()

    dense = model.depth_model

    # Apply ViT pos-enc patch (bilinear, no antialias) — done outside the temp
    # patcher because we want it to stay applied for the actual export call.
    _patch_vit_interpolate_pos_encoding(dense)
    _patch_decoder_get_bins(dense.decoder, device)

    wrapper = SafeMetric3DOnnxWrapper(dense).to(device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    dummy = torch.randn(1, 3, args.height, args.width, device=device, dtype=torch.float32)

    out_path = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with _export_patches(verbose=True):
        # warm up so dynamically registered buffers (depth_expectation_anchor)
        # are present during tracing
        _prewarm_decoder_buffers(wrapper, dummy)

        dynamic_axes = None
        if args.dynamic_batch:
            dynamic_axes = {"image": {0: "batch"}}
            for n in SafeMetric3DOnnxWrapper.output_names:
                dynamic_axes[n] = {0: "batch"}

        export_kwargs = dict(
            export_params=True,
            opset_version=args.opset,
            # Disabled: PyTorch 1.x constant-folding pass requires all tensors
            # to already be on the same device.  coords_grid() creates CPU zeros
            # that are later .to(cuda), which trips the fold-time device check.
            # TensorRT and ONNXRuntime both do their own constant folding at
            # engine-build / session-init time, so nothing is lost here.
            do_constant_folding=False,
            input_names=["image"],
            output_names=list(SafeMetric3DOnnxWrapper.output_names),
            dynamic_axes=dynamic_axes,
        )

        # `dynamo` kwarg only exists in PyTorch >= 2.x; probe once and pick the
        # right call path rather than relying on a live exception.
        _torch_major = int(torch.__version__.split(".")[0])
        if _torch_major >= 2:
            torch.onnx.export(wrapper, (dummy,), out_path, dynamo=False, **export_kwargs)
        else:
            torch.onnx.export(wrapper, (dummy,), out_path, **export_kwargs)

    print(f"[ok] saved ONNX to {out_path}")
    print(f"     T (RAFT iters baked into graph): {dense.decoder.iters}")
    print(f"     input  image:       float32 [{1 if not args.dynamic_batch else 'B'},3,{args.height},{args.width}]")
    print(f"     output pred_depth:  float32 [B,1,{args.height},{args.width}]")
    print(f"     output confidence:  float32 [B,1,{args.height},{args.width}]")
    print(f"     output pred_normal: float32 [B,4,{args.height},{args.width}]")
    print(f"     output safe_logits: float32 [B,2,{args.height},{args.width}]")
    print(f"     output safe_prob:   float32 [B,1,{args.height},{args.width}]")

    if not args.no_simplify:
        _maybe_simplify(out_path)


def _maybe_simplify(path: str) -> None:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("[simplify] onnx-simplifier not installed; skipping (pip install onnxsim)")
        return
    try:
        model = onnx.load(path)
        model_simp, ok = simplify(model)
        if ok:
            onnx.save(model_simp, path)
            print(f"[simplify] simplified ONNX written back to {path}")
        else:
            print("[simplify] onnx-simplifier reported it could not validate the simplified model; original kept")
    except Exception as exc:
        print(f"[simplify] failed: {exc}")


if __name__ == "__main__":
    main()
