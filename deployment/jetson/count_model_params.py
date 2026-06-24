#!/usr/bin/env python3
"""
统计 Safe Metric3D **原始 PyTorch 模型**参数量（与 safe_pre_demo / 训练一致）。

统计对象：`get_configured_monodepth_model(cfg)` → `depth_model`（DensePredModel：
encoder + RAFTDepthNormalSafeDPT5 decoder）。**不包含** ONNX 导出用的 wrapper、
也不包含部署时额外的 mean/std buffer。

Usage:
    cd <repo_root>
    python deployment/jetson/count_model_params.py

    python deployment/jetson/count_model_params.py \\
        --cfg mono/configs/HourglassDecoder/vit.raft5.small_uav_safe.py \\
        --ckpt final_model/student_step00004400_86.04.pth \\
        --detail
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

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
        print(f"  missing ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")


def _human(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f} G"
    if n >= 1_000_000:
        return f"{n / 1e6:.3f} M"
    if n >= 1_000:
        return f"{n / 1e3:.3f} K"
    return str(n)


def _mb(num_elements: int, bytes_per_elem: int = 4) -> float:
    return num_elements * bytes_per_elem / (1024.0 * 1024.0)


@dataclass
class ParamStats:
    name: str
    params: int = 0
    trainable: int = 0
    buffers: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params_human"] = _human(self.params)
        d["trainable_human"] = _human(self.trainable)
        d["buffers_human"] = _human(self.buffers)
        d["fp32_mb"] = round(_mb(self.params), 3)
        d["fp16_mb"] = round(_mb(self.params, 2), 3)
        return d


def count_module(module: nn.Module, name: str) -> ParamStats:
    params = trainable = buffers = 0
    for p in module.parameters():
        n = p.numel()
        params += n
        if p.requires_grad:
            trainable += n
    for b in module.buffers():
        buffers += b.numel()
    return ParamStats(name=name, params=params, trainable=trainable, buffers=buffers)


def count_decoder_children(decoder: nn.Module) -> List[ParamStats]:
    return [count_module(child, name) for name, child in decoder.named_children()]


def build_model(cfg_path: str, ckpt: Optional[str], iters: Optional[int], strict: bool) -> tuple:
    cfg = Config.fromfile(cfg_path)
    if iters is not None:
        cfg.model.decode_head.iters = int(iters)
    model = get_configured_monodepth_model(cfg)
    if ckpt:
        _load_state_dict(model, ckpt, strict=strict)
    model.eval()
    return model, cfg


def analyze_original(model: nn.Module, *, detail: bool) -> dict:
    """Original PyTorch stack: DepthModel → depth_model → encoder / decoder."""
    dense = model.depth_model
    rows: List[ParamStats] = [
        count_module(model, "DepthModel (wrapper, 原始入口)"),
        count_module(dense, "depth_model (DensePredModel)"),
    ]
    if hasattr(dense, "encoder"):
        rows.append(count_module(dense.encoder, "encoder (ViT backbone)"))
    if hasattr(dense, "decoder"):
        dec = dense.decoder
        rows.append(count_module(dec, "decoder (RAFTDepthNormalSafeDPT5)"))
        if detail:
            rows.extend(count_decoder_children(dec))

    summary = rows[1]  # depth_model is the actual network
    return {
        "summary": summary.to_dict(),
        "breakdown": [r.to_dict() for r in rows],
    }


def print_table(rows: List[ParamStats]) -> None:
    print(f"\n{'module':<44} {'params':>12} {'trainable':>12} {'buffers':>10} {'fp32 MB':>9}")
    print("-" * 91)
    for s in rows:
        print(
            f"{s.name:<44} {_human(s.params):>12} {_human(s.trainable):>12} "
            f"{_human(s.buffers):>10} {_mb(s.params):>9.2f}"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Count parameters of the original Safe Metric3D PyTorch model"
    )
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
        default=None,
        help="Override decode_head.iters (default: use value in cfg). Param count does not change with T.",
    )
    p.add_argument("--no-ckpt", action="store_true", help="Architecture only, do not load checkpoint")
    p.add_argument("--strict-state-dict", action="store_true")
    p.add_argument("--detail", action="store_true", help="List each first-level submodule inside decoder")
    p.add_argument("--json", type=str, default=None, help="Save report as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = args.cfg if os.path.isabs(args.cfg) else os.path.join(_REPO_ROOT, args.cfg)
    ckpt = None if args.no_ckpt else args.ckpt
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(_REPO_ROOT, ckpt)

    model, cfg = build_model(cfg_path, ckpt, args.iters, args.strict_state_dict)
    dense = model.depth_model
    iters = getattr(dense.decoder, "iters", args.iters or cfg.model.decode_head.iters)

    print("[target] original PyTorch model (NOT ONNX export wrapper)")
    print(f"[cfg   ] {cfg_path}")
    print(f"[type  ] backbone={cfg.model.backbone.type}  decode_head={cfg.model.decode_head.type}")
    print(f"[iters ] decode_head.iters={iters}  (T 只影响前向迭代次数，不改变参数量)")
    if ckpt:
        print(f"[ckpt  ] {ckpt}")
    else:
        print("[ckpt  ] (skipped)")

    rep = analyze_original(model, detail=args.detail)
    rows = [ParamStats(**{k: v for k, v in item.items() if k in ("name", "params", "trainable", "buffers")})
            for item in rep["breakdown"]]

    print_table(rows)

    s = rep["summary"]
    print(
        f"\n[原始网络 depth_model] "
        f"params={s['params']:,} ({s['params_human']})  "
        f"trainable={s['trainable']:,}  "
        f"buffers={s['buffers']:,}  "
        f"weights ≈ {s['fp32_mb']:.2f} MB (fp32) / {s['fp16_mb']:.2f} MB (fp16)"
    )

    if args.json:
        out_path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = {
            "target": "original_pytorch_depth_model",
            "cfg": cfg_path,
            "ckpt": ckpt,
            "decode_head_iters": iters,
            "model_type": {
                "backbone": cfg.model.backbone.type,
                "decode_head": cfg.model.decode_head.type,
            },
            **rep,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[json] {out_path}")


if __name__ == "__main__":
    main()
