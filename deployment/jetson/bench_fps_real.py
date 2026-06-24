#!/usr/bin/env python3
"""
=== Jetson side ===  End-to-end FPS benchmark on REAL images.

Given a directory of .onnx files (produced by deployment/jetson/prepare_bench.sh)
and a directory of test images, this script

  1. (re)builds a TensorRT engine for every (precision, T) in the grid
     (default: fp32, fp16  x  T=1, T=4),
  2. loops over the images, runs

        preprocess (cv2/numpy)  ->  H2D + execute_async_v3 + D2H  ->  postprocess

     and records timing for each stage,
  3. prints a markdown table and saves a CSV.

Engines are cached as `safe_metric3d_T{T}_{prec}.plan` next to the ONNX, so
rerunning the script reuses them unless `--rebuild` is given.

The TensorRT runtime is reused from `infer_tensorrt.py` (TrtRunner), so this
script has no dependency on PyTorch.

Usage:
    python3 bench_fps_real.py <onnx_dir> <images_dir>
    python3 bench_fps_real.py ./bench_models ~/datasets/wilduav/seq00/img \
        --iters 1 4 --precisions fp32 fp16 --repeat 2 --postprocess
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# These two modules already live in deployment/jetson/, they have no torch deps.
from infer_tensorrt import TrtRunner  # noqa: E402
from pipeline import (  # noqa: E402
    INPUT_SIZE_VIT,
    postprocess_depth,
    postprocess_normal,
    postprocess_safe,
    preprocess,
)


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_ITERS = (1, 4)
DEFAULT_PRECISIONS = ("fp32", "fp16")

TRTEXEC_CANDIDATES = (
    shutil.which("trtexec") or "",
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/tensorrt/bin/trtexec",
)


def find_trtexec(explicit: Optional[str]) -> str:
    if explicit:
        if not Path(explicit).is_file():
            sys.exit(f"--trtexec not found at: {explicit}")
        return explicit
    for c in TRTEXEC_CANDIDATES:
        if c and Path(c).is_file():
            return c
    sys.exit(
        "trtexec not found. Pass --trtexec /path/to/trtexec, or install TensorRT.\n"
        "On Jetson: /usr/src/tensorrt/bin/trtexec"
    )


def build_engine_if_missing(
    trtexec: str,
    onnx: Path,
    plan: Path,
    *,
    fp16: bool,
    workspace_mb: int,
    rebuild: bool,
) -> float:
    """Return seconds elapsed (0 if reused). Exit on failure."""
    if plan.exists() and not rebuild:
        print(f"[engine] reuse  : {plan.name}")
        return 0.0

    cmd = [trtexec, f"--onnx={onnx}", f"--saveEngine={plan}", "--buildOnly"]
    if fp16:
        cmd.append("--fp16")
    help_out = subprocess.run([trtexec, "--help"], capture_output=True, text=True).stdout
    if "--memPoolSize" in help_out:
        cmd.append(f"--memPoolSize=workspace:{workspace_mb}")
    else:
        cmd.append(f"--workspace={workspace_mb}")
    if "--builderOptimizationLevel" in help_out:
        cmd.append("--builderOptimizationLevel=5")

    print(f"[engine] build  : {plan.name}")
    print(f"         $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(log.splitlines()[-20:])
        sys.exit(f"trtexec build failed (rc={proc.returncode}):\n{tail}")
    print(f"         done in {elapsed:.1f}s")
    return elapsed


@dataclass
class Stage:
    samples: List[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def stats(self) -> Tuple[float, float, float, float]:
        if not self.samples:
            return (float("nan"),) * 4
        m = float(np.mean(self.samples))
        med = float(np.median(self.samples))
        p99 = float(np.percentile(self.samples, 99))
        s = float(np.std(self.samples))
        return m, med, p99, s


@dataclass
class BenchResult:
    precision: str
    T: int
    plan: str
    num_images: int
    repeats: int
    preprocess: Stage = field(default_factory=Stage)
    gpu: Stage = field(default_factory=Stage)
    postprocess: Stage = field(default_factory=Stage)
    total: Stage = field(default_factory=Stage)

    def fps(self) -> float:
        m, *_ = self.total.stats()
        return 1000.0 / m if m and not np.isnan(m) else float("nan")


def list_images(images_dir: Path, max_images: Optional[int]) -> List[Path]:
    files: List[Path] = []
    for ext in SUPPORTED_EXTS:
        files.extend(sorted(images_dir.glob(f"*{ext}")))
        files.extend(sorted(images_dir.glob(f"*{ext.upper()}")))
    files = sorted(set(files))
    if not files:
        sys.exit(f"no images found in {images_dir} (extensions: {SUPPORTED_EXTS})")
    if max_images and max_images > 0:
        files = files[:max_images]
    return files


def run_bench(
    plan: Path,
    *,
    images: List[Path],
    canonical_hw: Tuple[int, int],
    repeats: int,
    warmup_images: int,
    do_postprocess: bool,
    threshold: float,
    precision: str,
    T: int,
) -> BenchResult:
    runner = TrtRunner(str(plan))
    in_name = runner.input_names[0]
    runner.prepare({in_name: (1, 3, canonical_hw[0], canonical_hw[1])})

    res = BenchResult(
        precision=precision,
        T=T,
        plan=str(plan),
        num_images=len(images),
        repeats=repeats,
    )

    # Detect input dtype (fp16 IO support)
    target_dtype = runner.host[in_name].dtype

    # ---- warmup ----
    for i, img_path in enumerate(images[: max(1, warmup_images)]):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        x, _ = preprocess(bgr, intrinsic=None, is_bgr=True, canonical_hw=canonical_hw)
        if x.dtype != target_dtype:
            x = x.astype(target_dtype, copy=False)
        runner.infer({in_name: x})

    # ---- measure ----
    for rep in range(repeats):
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[warn] cannot read {img_path}, skipping")
                continue

            t_total0 = time.perf_counter()

            t0 = time.perf_counter()
            x, meta = preprocess(bgr, intrinsic=None, is_bgr=True, canonical_hw=canonical_hw)
            if x.dtype != target_dtype:
                x = x.astype(target_dtype, copy=False)
            t_pre = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            outs = runner.infer({in_name: x})
            t_gpu = (time.perf_counter() - t0) * 1000.0

            t_post = 0.0
            if do_postprocess:
                t0 = time.perf_counter()
                pred_depth = outs.get("pred_depth")
                safe_logits = outs.get("safe_logits")
                pred_normal = outs.get("pred_normal")
                if pred_depth is not None:
                    _ = postprocess_depth(pred_depth.astype(np.float32), meta, use_metric=False)
                if safe_logits is not None:
                    _ = postprocess_safe(safe_logits.astype(np.float32), meta, threshold=threshold)
                if pred_normal is not None:
                    _ = postprocess_normal(pred_normal.astype(np.float32), meta)
                t_post = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_total0) * 1000.0

            res.preprocess.add(t_pre)
            res.gpu.add(t_gpu)
            res.postprocess.add(t_post)
            res.total.add(t_total)

    del runner
    return res


def format_md(rows: List[BenchResult], do_postprocess: bool) -> str:
    if do_postprocess:
        header = (
            "| precision | T | preproc mean (ms) | gpu mean (ms) | postproc mean (ms) | "
            "e2e mean (ms) | e2e median (ms) | e2e p99 (ms) | FPS (e2e mean) | FPS (gpu only) |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|"
    else:
        header = (
            "| precision | T | preproc mean (ms) | gpu mean (ms) | "
            "e2e mean (ms) | e2e median (ms) | e2e p99 (ms) | FPS (e2e mean) | FPS (gpu only) |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|"

    lines = [header, sep]
    for r in rows:
        pre_m, *_ = r.preprocess.stats()
        gpu_m, *_ = r.gpu.stats()
        post_m, *_ = r.postprocess.stats()
        tot_m, tot_med, tot_p99, _ = r.total.stats()
        fps_total = 1000.0 / tot_m if tot_m else float("nan")
        fps_gpu = 1000.0 / gpu_m if gpu_m else float("nan")
        cells = [
            r.precision,
            str(r.T),
            f"{pre_m:.2f}",
            f"{gpu_m:.2f}",
        ]
        if do_postprocess:
            cells.append(f"{post_m:.2f}")
        cells += [
            f"{tot_m:.2f}",
            f"{tot_med:.2f}",
            f"{tot_p99:.2f}",
            f"{fps_total:.2f}",
            f"{fps_gpu:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_csv(rows: List[BenchResult], path: Path, do_postprocess: bool) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "precision",
            "T",
            "num_images",
            "repeats",
            "preprocess_mean_ms",
            "preprocess_median_ms",
            "preprocess_p99_ms",
            "gpu_mean_ms",
            "gpu_median_ms",
            "gpu_p99_ms",
        ]
        if do_postprocess:
            header += ["postprocess_mean_ms", "postprocess_median_ms", "postprocess_p99_ms"]
        header += [
            "e2e_mean_ms",
            "e2e_median_ms",
            "e2e_p99_ms",
            "fps_e2e_mean",
            "fps_gpu_only",
            "plan",
        ]
        w.writerow(header)
        for r in rows:
            pre_m, pre_med, pre_p99, _ = r.preprocess.stats()
            gpu_m, gpu_med, gpu_p99, _ = r.gpu.stats()
            post_m, post_med, post_p99, _ = r.postprocess.stats()
            tot_m, tot_med, tot_p99, _ = r.total.stats()
            row = [
                r.precision,
                r.T,
                r.num_images,
                r.repeats,
                f"{pre_m:.4f}",
                f"{pre_med:.4f}",
                f"{pre_p99:.4f}",
                f"{gpu_m:.4f}",
                f"{gpu_med:.4f}",
                f"{gpu_p99:.4f}",
            ]
            if do_postprocess:
                row += [f"{post_m:.4f}", f"{post_med:.4f}", f"{post_p99:.4f}"]
            row += [
                f"{tot_m:.4f}",
                f"{tot_med:.4f}",
                f"{tot_p99:.4f}",
                f"{r.fps():.4f}",
                f"{1000.0 / gpu_m:.4f}" if gpu_m else "",
                r.plan,
            ]
            w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end FPS benchmark on real images")
    p.add_argument("models_dir", help="directory containing safe_metric3d_T*.onnx (and built .plan files)")
    p.add_argument("images_dir", help="directory of real images to feed through the model")
    p.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERS),
        help=f"T values (default: {DEFAULT_ITERS})",
    )
    p.add_argument(
        "--precisions",
        nargs="+",
        choices=["fp32", "fp16"],
        default=list(DEFAULT_PRECISIONS),
        help=f"precisions (default: {DEFAULT_PRECISIONS})",
    )
    p.add_argument("--height", type=int, default=INPUT_SIZE_VIT[0])
    p.add_argument("--width", type=int, default=INPUT_SIZE_VIT[1])
    p.add_argument("--repeat", type=int, default=1, help="number of passes over the whole image list")
    p.add_argument("--warmup-images", type=int, default=5, help="images used purely for warmup (not timed)")
    p.add_argument("--max-images", type=int, default=0, help="cap number of images (0 = all)")
    p.add_argument("--postprocess", action="store_true", help="include postprocess time in the e2e budget")
    p.add_argument("--threshold", type=float, default=0.5, help="safe-prob threshold for postprocess")
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--trtexec", type=str, default=None)
    p.add_argument("--rebuild", action="store_true", help="rebuild .plan engines even if they already exist")
    p.add_argument("--csv", type=str, default=None, help="output CSV (default: <models_dir>/bench_fps_real.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    models_dir = Path(args.models_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    if not models_dir.is_dir():
        sys.exit(f"models_dir not found: {models_dir}")
    if not images_dir.is_dir():
        sys.exit(f"images_dir not found: {images_dir}")

    images = list_images(images_dir, args.max_images)
    print(f"[bench] {len(images)} image(s) from {images_dir}")
    print(f"[bench] grid: T={args.iters}  precisions={args.precisions}  repeat={args.repeat}")

    trtexec = find_trtexec(args.trtexec)

    # sanity-check ONNX availability
    for T in args.iters:
        onnx = models_dir / f"safe_metric3d_T{T}.onnx"
        if not onnx.is_file():
            sys.exit(f"missing ONNX for T={T}: {onnx}\nRun prepare_bench.sh first.")

    # Build engines first (so we can isolate engine-build time from inference timing)
    for T in args.iters:
        onnx = models_dir / f"safe_metric3d_T{T}.onnx"
        for prec in args.precisions:
            plan = models_dir / f"safe_metric3d_T{T}_{prec}.plan"
            build_engine_if_missing(
                trtexec,
                onnx,
                plan,
                fp16=(prec == "fp16"),
                workspace_mb=args.workspace_mb,
                rebuild=args.rebuild,
            )

    # Run benches
    rows: List[BenchResult] = []
    for T in args.iters:
        for prec in args.precisions:
            plan = models_dir / f"safe_metric3d_T{T}_{prec}.plan"
            print(f"\n[run] precision={prec}  T={T}")
            res = run_bench(
                plan,
                images=images,
                canonical_hw=(args.height, args.width),
                repeats=args.repeat,
                warmup_images=args.warmup_images,
                do_postprocess=args.postprocess,
                threshold=args.threshold,
                precision=prec,
                T=T,
            )
            rows.append(res)

            pre_m, *_ = res.preprocess.stats()
            gpu_m, *_ = res.gpu.stats()
            tot_m, tot_med, tot_p99, _ = res.total.stats()
            post_m, *_ = res.postprocess.stats() if args.postprocess else (0.0, 0.0, 0.0, 0.0)
            print(
                f"      preproc={pre_m:6.2f} ms | gpu={gpu_m:6.2f} ms"
                + (f" | postproc={post_m:6.2f} ms" if args.postprocess else "")
                + f" | e2e mean={tot_m:6.2f} ms (median={tot_med:.2f}, p99={tot_p99:.2f})"
                + f" | FPS={res.fps():.2f}"
            )

    md = format_md(rows, args.postprocess)
    print("\n" + "=" * 80)
    print(md)
    print("=" * 80)

    csv_path = Path(args.csv) if args.csv else models_dir / "bench_fps_real.csv"
    save_csv(rows, csv_path, args.postprocess)
    md_path = csv_path.with_suffix(".md")
    md_path.write_text(md + "\n")
    print(f"\n[csv] {csv_path}")
    print(f"[md ] {md_path}")


if __name__ == "__main__":
    main()
