#!/usr/bin/env python3
"""
统计 Safe Metric3D **原始 PyTorch 模型** FLOPs（与 count_model_params.py 同一套 cfg/ckpt）。

统计对象：`depth_model`（DensePredModel = encoder + decoder），输入为 eval 用的
ImageNet 归一化张量 [1,3,H,W]。FLOPs 会随 `decode_head.iters`（RAFT 迭代 T）变化。

依赖（主机上推荐 thop；fvcore 对 ViT+RAFT 的 JIT trace 常会失败或严重漏计）:
    pip install thop
    pip install fvcore   # 可选，--backend fvcore

Usage:
    cd <repo_root>
    python deployment/jetson/count_model_flops.py
    python deployment/jetson/count_model_flops.py --iters 1 4 --backend thop
    python deployment/jetson/count_model_flops.py --height 616 --width 1064 --json flops.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from mmengine import Config
except ImportError:
    from mmcv import Config  # type: ignore

from mono.model.monodepth_model import get_configured_monodepth_model

INPUT_SIZE_VIT = (616, 1064)  # (H, W)


def _human_flops(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.3f} MFLOPs"
    if n >= 1e3:
        return f"{n / 1e3:.3f} KFLOPs"
    return f"{n:.0f}"


def _load_state_dict(model: nn.Module, ckpt_path: str, strict: bool) -> None:
    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict) and "model_state_dict" in blob:
        sd = blob["model_state_dict"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    else:
        sd = blob
    model.load_state_dict(sd, strict=strict)


def build_model(cfg_path: str, ckpt: Optional[str], iters: int, strict: bool) -> Tuple[nn.Module, object]:
    cfg = Config.fromfile(cfg_path)
    cfg.model.decode_head.iters = int(iters)
    model = get_configured_monodepth_model(cfg)
    if ckpt:
        _load_state_dict(model, ckpt, strict=strict)
    model.eval()
    return model, cfg


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return dev


def _patch_decoder_get_bins(decoder: nn.Module, device: torch.device) -> None:
    """Decoder.get_bins hard-codes device='cuda'; align with profiling device."""
    import math

    def get_bins(self, bins_num):
        depth_bins_vec = torch.linspace(
            math.log(self.min_val), math.log(self.max_val), bins_num, device=device
        )
        return torch.exp(depth_bins_vec)

    decoder.get_bins = get_bins.__get__(decoder, decoder.__class__)  # type: ignore[assignment]


def _decoder_source_modules() -> list:
    """Decoder files do `from torch.utils.checkpoint import checkpoint` — patch the module symbol."""
    mods = []
    for name in (
        "mono.model.decode_heads.RAFTDepthNormalSafeDPTDecoder5_bestbak",
        "mono.model.decode_heads.RAFTDepthNormalDPTDecoder5_new",
    ):
        try:
            import importlib

            mods.append(importlib.import_module(name))
        except Exception:
            pass
    return mods


@contextlib.contextmanager
def _profile_runtime_patches(decoder: nn.Module):
    """
    Patches needed for a successful forward under fvcore/thop tracing:
      - checkpoint passthrough (global + decoder module imports)
      - copy.deepcopy -> tensor.clone (RAFT safe_net0 init)
      - torch.isnan/isinf no-ops (avoid tracer bool issues)
    """
    import copy as copy_mod
    import torch.utils.checkpoint as ckpt_mod

    saved_ckpt = ckpt_mod.checkpoint

    def _noop_checkpoint(fn, *args, **kwargs):
        kwargs.pop("use_reentrant", None)
        kwargs.pop("preserve_rng_state", None)
        kwargs.pop("determinism_check", None)
        kwargs.pop("debug", None)
        return fn(*args)

    ckpt_mod.checkpoint = _noop_checkpoint

    saved_mod_ckpt = []
    for mod in _decoder_source_modules():
        if hasattr(mod, "checkpoint"):
            saved_mod_ckpt.append((mod, mod.checkpoint))
            mod.checkpoint = _noop_checkpoint  # type: ignore[attr-defined]

    class _AlwaysFalseBool:
        def any(self):
            return self

        def all(self):
            return self

        def __bool__(self):
            return False

    def _noop_isnan(*_a, **_k):
        return _AlwaysFalseBool()

    def _noop_isinf(*_a, **_k):
        return _AlwaysFalseBool()

    saved_isnan, saved_isinf = torch.isnan, torch.isinf
    torch.isnan = _noop_isnan  # type: ignore[assignment]
    torch.isinf = _noop_isinf  # type: ignore[assignment]

    orig_deepcopy = copy_mod.deepcopy
    dec_copy = getattr(decoder, "copy", None)

    def _tensor_safe_deepcopy(obj, memo=None):
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        return orig_deepcopy(obj, memo)

    copy_mod.deepcopy = _tensor_safe_deepcopy  # type: ignore[assignment]
    if dec_copy is not None and hasattr(dec_copy, "deepcopy"):
        saved_dec_deepcopy = dec_copy.deepcopy
        dec_copy.deepcopy = _tensor_safe_deepcopy  # type: ignore[attr-defined]
    else:
        saved_dec_deepcopy = None

    try:
        yield
    finally:
        ckpt_mod.checkpoint = saved_ckpt
        for mod, orig in saved_mod_ckpt:
            mod.checkpoint = orig  # type: ignore[attr-defined]
        torch.isnan = saved_isnan
        torch.isinf = saved_isinf
        copy_mod.deepcopy = orig_deepcopy
        if saved_dec_deepcopy is not None and dec_copy is not None:
            dec_copy.deepcopy = saved_dec_deepcopy


@contextlib.contextmanager
def _patch_xformers_off():
    """与 ONNX 导出一致：禁用 xformers，走标准 attention，便于 thop 统计。"""
    saved = {}
    try:
        import mono.model.backbones.ViT_DINO_reg as vitmod  # noqa: WPS433

        saved["XFORMERS_AVAILABLE"] = vitmod.XFORMERS_AVAILABLE
        vitmod.XFORMERS_AVAILABLE = False
    except Exception:
        pass
    try:
        yield
    finally:
        if "XFORMERS_AVAILABLE" in saved:
            import mono.model.backbones.ViT_DINO_reg as vitmod  # noqa: WPS433

            vitmod.XFORMERS_AVAILABLE = saved["XFORMERS_AVAILABLE"]


def _clear_thop_hooks(module: nn.Module) -> None:
    """thop.profile 会注册 forward hooks，多次调用前需清理。"""
    for m in module.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()
        if hasattr(m, "total_ops"):
            del m.total_ops
        if hasattr(m, "total_params"):
            del m.total_params


def prewarm_dense(dense: nn.Module, dummy: torch.Tensor) -> None:
    """Register runtime buffers (e.g. depth_expectation_anchor) on the correct device."""
    _patch_decoder_get_bins(dense.decoder, dummy.device)
    with torch.no_grad():
        _ = dense(dummy)


def make_dummy_input(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Eval 路径下 depth_model 接收已 ImageNet 归一化的 NCHW；FLOPs 与数值无关。"""
    return torch.randn(1, 3, height, width, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def count_flops_fvcore(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> int:
    from fvcore.nn import FlopCountAnalysis

    with torch.no_grad():
        analysis = FlopCountAnalysis(module, inputs)
        total = int(analysis.total())
        # fvcore requires total() before unsupported_ops()
        try:
            unsupported = analysis.unsupported_ops()
            if unsupported:
                print(
                    f"[warn] fvcore unsupported ops ({len(unsupported)} types), "
                    "FLOPs may be underestimated"
                )
        except (RuntimeError, AttributeError):
            pass
        return total


def count_flops_thop(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> int:
    from thop import profile

    _clear_thop_hooks(module)
    with torch.no_grad():
        macs, _ = profile(module, inputs=inputs, verbose=False)
    _clear_thop_hooks(module)
    # thop 报告 MACs；1 MAC ≈ 2 FLOPs（一次乘加）
    return int(macs * 2)


def _count_with_fallback(
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    backend: str,
) -> Tuple[int, str]:
    """返回 (flops, backend_used)。"""
    if backend == "thop":
        return count_flops_thop(module, inputs), "thop"
    try:
        return count_flops_fvcore(module, inputs), "fvcore"
    except RuntimeError as exc:
        print(f"[warn] fvcore trace failed on {module.__class__.__name__}: {exc}")
        print("[warn] falling back to thop for this module")
        return count_flops_thop(module, inputs), "thop"


def profile_dense(
    dense: nn.Module,
    dummy: torch.Tensor,
    backend: str,
) -> Dict[str, int]:
    x = (dummy,)
    total, _ = _count_with_fallback(dense, x, backend)

    enc_flops, _ = _count_with_fallback(dense.encoder, x, backend)
    with torch.no_grad():
        feats = dense.encoder(dummy)
    dec_flops, _ = _count_with_fallback(dense.decoder, (feats,), backend)

    return {
        "depth_model_total": total,
        "encoder": enc_flops,
        "decoder": dec_flops,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Count FLOPs of original Safe Metric3D PyTorch model")
    p.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(_REPO_ROOT, "mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py"),
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=os.environ.get(
            "CKPT",
            os.path.join(_REPO_ROOT, "final_model", "student_step00004400_86.04.pth"),
        ),
    )
    p.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=None,
        help="RAFT iters T to profile (default: cfg value only)",
    )
    p.add_argument("--height", type=int, default=INPUT_SIZE_VIT[0])
    p.add_argument("--width", type=int, default=INPUT_SIZE_VIT[1])
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--strict-state-dict", action="store_true")
    p.add_argument(
        "--backend",
        choices=["thop", "fvcore"],
        default="thop",
        help="thop=hook 统计(推荐); fvcore=JIT trace(ViT 易失败/漏计)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cuda | cpu (default: cuda if available, matches training)",
    )
    p.add_argument("--json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = args.cfg if os.path.isabs(args.cfg) else os.path.join(_REPO_ROOT, args.cfg)
    ckpt = None if args.no_ckpt else args.ckpt
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(_REPO_ROOT, ckpt)

    cfg_probe = Config.fromfile(cfg_path)
    iters_list = args.iters if args.iters else [int(cfg_probe.model.decode_head.iters)]

    device = resolve_device(args.device)
    dummy = make_dummy_input(args.height, args.width, device)

    print("[target] original PyTorch depth_model FLOPs (NOT ONNX wrapper)")
    print(f"[cfg   ] {cfg_path}")
    print(f"[input ] [1,3,{args.height},{args.width}]  normalized (eval path)")
    print(f"[backend] {args.backend}  device={device}")
    if ckpt:
        print(f"[ckpt  ] {ckpt}")
    else:
        print("[ckpt  ] (skipped)")

    reports = []
    print(f"\n{'T':>3}  {'encoder':>14}  {'decoder':>14}  {'total':>14}  {'decoder/T':>14}")
    print("-" * 65)

    for T in iters_list:
        model, _cfg = build_model(cfg_path, ckpt, T, args.strict_state_dict)
        dense = model.depth_model.to(device)
        with _patch_xformers_off(), _profile_runtime_patches(dense.decoder):
            prewarm_dense(dense, dummy)
            stats = profile_dense(dense, dummy, args.backend)
            dec_per_iter = stats["decoder"] / max(T, 1)
            row = {
                "decode_head_iters": T,
                "input_shape": [1, 3, args.height, args.width],
                "backend": args.backend,
                "flops": {k: v for k, v in stats.items()},
                "flops_human": {k: _human_flops(v) for k, v in stats.items()},
                "decoder_flops_per_iter_estimate": int(dec_per_iter),
                "decoder_flops_per_iter_human": _human_flops(dec_per_iter),
            }
            reports.append(row)
            print(
                f"{T:>3}  "
                f"{_human_flops(stats['encoder']):>14}  "
                f"{_human_flops(stats['decoder']):>14}  "
                f"{_human_flops(stats['depth_model_total']):>14}  "
                f"{_human_flops(dec_per_iter):>14}"
            )
        del model, dense

    print(
        "\n[note] FLOPs 为理论乘加次数估计（thop: MACs×2）。"
        "encoder+decoder 分项之和可能与 total 略有出入（thop hook 统计方式）。"
    )
    print("[note] fvcore 对 ViT/RAFT 常报 unsupported ops 或 trace 失败，主机上请用 --backend thop。")
    print("[note] decoder/T 仅为 decoder÷T 的粗算，实际每步算子组合略有差异。")
    print("[note] 部署 ONNX/TRT 若 T=4，计算量量级与表中 T=4 的 total 接近。")

    if args.json:
        out_path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = {
            "target": "original_pytorch_depth_model_flops",
            "cfg": cfg_path,
            "ckpt": ckpt,
            "model_type": {
                "backbone": cfg_probe.model.backbone.type,
                "decode_head": cfg_probe.model.decode_head.type,
            },
            "reports": reports,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[json] {out_path}")


if __name__ == "__main__":
    main()
