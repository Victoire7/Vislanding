#!/usr/bin/env python3
"""
Convert Safe Metric3D ONNX from FP32 to FP16.

Two modes are supported:

    * default — `keep_io_types=True`. The graph weights/activations run in
      float16 internally, but the network's inputs/outputs stay float32, which
      means the rest of your inference pipeline does not have to change dtype.
      This is the recommended mode for most Jetson deployments.

    * --full-fp16 — also casts inputs/outputs to float16, useful when you want
      to feed half-precision tensors directly (saves a copy/cast).

You can also keep specific nodes in fp32 to mitigate accuracy loss on the
"sensitive" math (LayerNorm, Softmax). Use --keep-fp32-ops to add op types.

Optional: --simplify runs `onnxsim.simplify` after FP16 conversion (recommended).

Install:
    pip install onnx onnxconverter-common onnxsim
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--full-fp16",
        action="store_true",
        help="Also cast graph IO to float16 (default: keep IO float32)",
    )
    p.add_argument(
        "--keep-fp32-ops",
        type=str,
        nargs="*",
        default=[
            # Numerically sensitive ops kept in float32 for stability.
            "LayerNormalization",
            "InstanceNormalization",
            "GroupNormalization",
        ],
        help="Op types whose outputs should remain float32",
    )
    p.add_argument("--simplify", action="store_true", help="run onnxsim afterwards")
    return p.parse_args()


def _convert(model, *, keep_io_types: bool, op_block_list: Optional[List[str]] = None):
    from onnxconverter_common import float16 as oc_float16

    return oc_float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        op_block_list=op_block_list or [],
        disable_shape_infer=False,
    )


def main():
    args = parse_args()
    try:
        import onnx
    except ImportError:
        sys.exit("onnx is required: pip install onnx")
    try:
        import onnxconverter_common  # noqa: F401
    except ImportError:
        sys.exit("onnxconverter-common is required: pip install onnxconverter-common")

    if not os.path.isfile(args.input):
        sys.exit(f"input not found: {args.input}")

    model = onnx.load(args.input)
    print(f"[fp16] loaded {args.input} (ir={model.ir_version}, opset={model.opset_import[0].version})")

    model_fp16 = _convert(
        model,
        keep_io_types=not args.full_fp16,
        op_block_list=args.keep_fp32_ops,
    )

    out = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    onnx.save(model_fp16, out)
    print(f"[fp16] wrote {out}  io={'float16' if args.full_fp16 else 'float32'}  block={args.keep_fp32_ops}")

    if args.simplify:
        try:
            from onnxsim import simplify
        except ImportError:
            print("[simplify] onnxsim not installed; skipping (pip install onnxsim)")
            return
        try:
            simplified, ok = simplify(onnx.load(out))
            if ok:
                onnx.save(simplified, out)
                print(f"[simplify] {out} simplified")
            else:
                print("[simplify] simplifier could not validate output; original kept")
        except Exception as exc:
            print(f"[simplify] failed: {exc}")


if __name__ == "__main__":
    main()
