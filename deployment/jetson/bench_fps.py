#!/usr/bin/env python3
"""
=== Jetson side ===  One-shot FPS benchmark for Safe Metric3D.

For every (precision, T) in the grid (default: fp32 x fp16  vs  T=1 x T=4) this
script

  1. builds a TensorRT engine from `safe_metric3d_T{T}.onnx` (re-uses cached .plan
     if it already exists and --rebuild is not given),
  2. runs trtexec with --noDataTransfers --useCudaGraph for clean GPU-compute
     timing,
  3. parses the per-config mean / median GPU compute time and converts to FPS,
  4. prints a markdown table to stdout and a CSV next to the engines.

It expects the ONNX files exported by deployment/jetson/prepare_bench.sh, e.g.

    safe_metric3d_T1.onnx
    safe_metric3d_T4.onnx

Usage:
    python bench_fps.py /path/to/models_dir
    python bench_fps.py /path/to/models_dir --iters 1 4 --precisions fp32 fp16
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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
        "trtexec not found. Pass --trtexec /path/to/trtexec, or install TensorRT. "
        "On Jetson it's at /usr/src/tensorrt/bin/trtexec."
    )


@dataclass
class BenchRow:
    precision: str
    T: int
    onnx: str
    plan: str
    build_seconds: float
    compute_mean_ms: Optional[float]
    compute_median_ms: Optional[float]
    latency_mean_ms: Optional[float]
    throughput_qps: Optional[float]

    @property
    def fps_mean(self) -> Optional[float]:
        return 1000.0 / self.compute_mean_ms if self.compute_mean_ms else None

    @property
    def fps_median(self) -> Optional[float]:
        return 1000.0 / self.compute_median_ms if self.compute_median_ms else None


_RE_GPU = re.compile(
    r"GPU Compute Time:.*?mean\s*=\s*([\d.]+)\s*ms.*?median\s*=\s*([\d.]+)\s*ms",
    re.S,
)
_RE_LAT = re.compile(
    r"Latency:.*?mean\s*=\s*([\d.]+)\s*ms",
    re.S,
)
_RE_TPS = re.compile(r"Throughput:\s*([\d.]+)\s*qps")


def parse_trtexec(log_text: str):
    gpu = _RE_GPU.search(log_text)
    lat = _RE_LAT.search(log_text)
    tps = _RE_TPS.search(log_text)
    return (
        float(gpu.group(1)) if gpu else None,
        float(gpu.group(2)) if gpu else None,
        float(lat.group(1)) if lat else None,
        float(tps.group(1)) if tps else None,
    )


def build_and_bench(
    trtexec: str,
    onnx: Path,
    plan: Path,
    log_path: Path,
    *,
    fp16: bool,
    warmup_ms: int,
    iterations: int,
    workspace_mb: int,
    extra_args: List[str],
) -> BenchRow:
    cmd = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={plan}",
        f"--warmUp={warmup_ms}",
        f"--iterations={iterations}",
        f"--avgRuns={iterations}",
        "--useCudaGraph",
        "--noDataTransfers",
    ]
    if fp16:
        cmd.append("--fp16")
    # detect newer/older workspace flag
    help_out = subprocess.run([trtexec, "--help"], capture_output=True, text=True).stdout
    if "--memPoolSize" in help_out:
        cmd.append(f"--memPoolSize=workspace:{workspace_mb}")
    else:
        cmd.append(f"--workspace={workspace_mb}")
    if "--builderOptimizationLevel" in help_out:
        cmd.append("--builderOptimizationLevel=5")
    cmd.extend(extra_args)

    print(f"\n[bench] precision={'fp16' if fp16 else 'fp32'}  T={_t_from_onnx(onnx)}")
    print(f"[bench] $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    log_text = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log_text)
    if proc.returncode != 0:
        tail = "\n".join(log_text.splitlines()[-20:])
        sys.exit(f"trtexec failed (rc={proc.returncode}). Tail of log:\n{tail}")

    mean_ms, median_ms, lat_mean_ms, tps = parse_trtexec(log_text)
    row = BenchRow(
        precision="fp16" if fp16 else "fp32",
        T=_t_from_onnx(onnx),
        onnx=str(onnx),
        plan=str(plan),
        build_seconds=elapsed,
        compute_mean_ms=mean_ms,
        compute_median_ms=median_ms,
        latency_mean_ms=lat_mean_ms,
        throughput_qps=tps,
    )
    return row


def _t_from_onnx(onnx_path: Path) -> int:
    m = re.search(r"_T(\d+)\.onnx$", onnx_path.name)
    if not m:
        sys.exit(f"Cannot infer T from filename: {onnx_path.name} (expected ..._T<N>.onnx)")
    return int(m.group(1))


def print_markdown(rows: List[BenchRow]) -> str:
    header = (
        "| precision | T | GPU compute mean (ms) | GPU compute median (ms) | "
        "FPS (mean) | FPS (median) | trtexec throughput (qps) | engine |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            "| {p} | {t} | {cmean} | {cmedian} | {fmean} | {fmedian} | {tps} | `{eng}` |".format(
                p=r.precision,
                t=r.T,
                cmean=f"{r.compute_mean_ms:.3f}" if r.compute_mean_ms else "n/a",
                cmedian=f"{r.compute_median_ms:.3f}" if r.compute_median_ms else "n/a",
                fmean=f"{r.fps_mean:.2f}" if r.fps_mean else "n/a",
                fmedian=f"{r.fps_median:.2f}" if r.fps_median else "n/a",
                tps=f"{r.throughput_qps:.2f}" if r.throughput_qps else "n/a",
                eng=Path(r.plan).name,
            )
        )
    return "\n".join(lines)


def write_csv(rows: List[BenchRow], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "precision",
                "T",
                "gpu_compute_mean_ms",
                "gpu_compute_median_ms",
                "fps_mean",
                "fps_median",
                "trtexec_throughput_qps",
                "trtexec_latency_mean_ms",
                "build_seconds",
                "onnx",
                "engine",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.precision,
                    r.T,
                    r.compute_mean_ms or "",
                    r.compute_median_ms or "",
                    f"{r.fps_mean:.4f}" if r.fps_mean else "",
                    f"{r.fps_median:.4f}" if r.fps_median else "",
                    r.throughput_qps or "",
                    r.latency_mean_ms or "",
                    f"{r.build_seconds:.1f}",
                    r.onnx,
                    r.plan,
                ]
            )


def parse_args():
    p = argparse.ArgumentParser(description="FPS benchmark for Safe Metric3D on Jetson")
    p.add_argument("models_dir", help="directory containing safe_metric3d_T*.onnx")
    p.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERS),
        help=f"T values to bench (default: {DEFAULT_ITERS})",
    )
    p.add_argument(
        "--precisions",
        nargs="+",
        choices=["fp32", "fp16"],
        default=list(DEFAULT_PRECISIONS),
        help=f"precisions (default: {DEFAULT_PRECISIONS})",
    )
    p.add_argument("--warmup-ms", type=int, default=int(os.environ.get("WARMUP_MS", 2000)))
    p.add_argument("--iterations", type=int, default=int(os.environ.get("RUN_ITERATIONS", 200)))
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--trtexec", type=str, default=None, help="explicit trtexec path")
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="re-build engines even if .plan already exists (default: skip build, run trtexec --loadEngine)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="output CSV path (default: <models_dir>/bench_fps.csv)",
    )
    p.add_argument(
        "extra",
        nargs="*",
        help="additional args forwarded to trtexec",
    )
    return p.parse_args()


def main():
    args = parse_args()
    trtexec = find_trtexec(args.trtexec)
    models_dir = Path(args.models_dir).resolve()
    if not models_dir.is_dir():
        sys.exit(f"models dir not found: {models_dir}")

    csv_path = Path(args.csv) if args.csv else models_dir / "bench_fps.csv"

    # sanity-check ONNX availability
    for T in args.iters:
        onnx = models_dir / f"safe_metric3d_T{T}.onnx"
        if not onnx.is_file():
            sys.exit(f"missing ONNX for T={T}: {onnx}\nRun prepare_bench.sh first.")

    rows: List[BenchRow] = []
    for T in args.iters:
        onnx = models_dir / f"safe_metric3d_T{T}.onnx"
        for prec in args.precisions:
            fp16 = prec == "fp16"
            plan = models_dir / f"safe_metric3d_T{T}_{prec}.plan"
            log = models_dir / f"safe_metric3d_T{T}_{prec}.log"

            if plan.exists() and not args.rebuild:
                print(f"\n[bench] reusing engine: {plan}")
                cmd = [
                    trtexec,
                    f"--loadEngine={plan}",
                    f"--warmUp={args.warmup_ms}",
                    f"--iterations={args.iterations}",
                    f"--avgRuns={args.iterations}",
                    "--useCudaGraph",
                    "--noDataTransfers",
                ]
                print(f"[bench] $ {' '.join(cmd)}")
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.time() - t0
                log_text = (proc.stdout or "") + (proc.stderr or "")
                log.write_text(log_text)
                if proc.returncode != 0:
                    sys.exit(f"trtexec --loadEngine failed: {log_text.splitlines()[-5:]}")
                mean_ms, median_ms, lat_mean_ms, tps = parse_trtexec(log_text)
                rows.append(
                    BenchRow(
                        precision=prec,
                        T=T,
                        onnx=str(onnx),
                        plan=str(plan),
                        build_seconds=elapsed,
                        compute_mean_ms=mean_ms,
                        compute_median_ms=median_ms,
                        latency_mean_ms=lat_mean_ms,
                        throughput_qps=tps,
                    )
                )
            else:
                rows.append(
                    build_and_bench(
                        trtexec,
                        onnx,
                        plan,
                        log,
                        fp16=fp16,
                        warmup_ms=args.warmup_ms,
                        iterations=args.iterations,
                        workspace_mb=args.workspace_mb,
                        extra_args=args.extra,
                    )
                )

    md = print_markdown(rows)
    print("\n" + "=" * 70)
    print(md)
    print("=" * 70)

    write_csv(rows, csv_path)
    print(f"\n[csv ] {csv_path}")
    md_path = csv_path.with_suffix(".md")
    md_path.write_text(md + "\n")
    print(f"[md  ] {md_path}")


if __name__ == "__main__":
    main()
