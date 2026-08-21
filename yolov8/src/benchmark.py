"""
Real-Time Inference Latency and FPS Benchmark for YOLOv8 Wildlife Subsystem
Executes rigorous GPU-synchronized benchmarking on the target hardware (NVIDIA RTX 4050).
"""

from typing import Dict, Any, List
import os
import sys
import time
import argparse
import numpy as np
import torch
from ultralytics import YOLO


def benchmark_model(
    model_path: str,
    imgsz: int = 640,
    device: int = 0,
    warmup_runs: int = 20,
    test_runs: int = 100
) -> Dict[str, Any]:
    """
    Measure hardware-accurate latency and FPS with torch.cuda.synchronize().
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(model_path)
    is_cuda = torch.cuda.is_available() and device >= 0
    dev_str = f"cuda:{device}" if is_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"

    print("=" * 70)
    print(f"BENCHMARKING MODEL: {os.path.basename(model_path)}")
    print(f"Target Hardware:   {gpu_name}")
    print(f"Input Resolution:  {imgsz}x{imgsz}")
    print(f"Warmup Iterations: {warmup_runs}")
    print(f"Timing Iterations: {test_runs}")
    print("=" * 70)

    # Synthetic random image simulating real camera frame
    rng = np.random.RandomState(42)
    sample_frame = rng.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    # 1. Warm up GPU to eliminate CUDA context & cache miss latency
    print("Warming up target device...")
    for _ in range(warmup_runs):
        _ = model.predict(source=sample_frame, device=dev_str, imgsz=imgsz, verbose=False)
    if is_cuda:
        torch.cuda.synchronize()

    # 2. Benchmark Loop with CUDA event timing
    inference_times = []
    
    print(f"Executing {test_runs} timed iterations...")
    for _ in range(test_runs):
        if is_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        _ = model.predict(source=sample_frame, device=dev_str, imgsz=imgsz, verbose=False)

        if is_cuda:
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        inference_times.append((t_end - t_start) * 1000.0)  # in milliseconds

    times_np = np.array(inference_times)
    mean_latency_ms = float(np.mean(times_np))
    std_latency_ms = float(np.std(times_np))
    min_latency_ms = float(np.min(times_np))
    max_latency_ms = float(np.max(times_np))
    p95_latency_ms = float(np.percentile(times_np, 95))
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    # Model size in MB
    model_size_mb = os.path.getsize(model_path) / (1024.0 * 1024.0)

    benchmark_summary = {
        "model_file": os.path.basename(model_path),
        "hardware": gpu_name,
        "resolution": f"{imgsz}x{imgsz}",
        "mean_latency_ms": round(mean_latency_ms, 2),
        "std_latency_ms": round(std_latency_ms, 2),
        "min_latency_ms": round(min_latency_ms, 2),
        "max_latency_ms": round(max_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "fps": round(fps, 2),
        "model_size_mb": round(model_size_mb, 2)
    }

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS:")
    print(f"  Mean Total Latency:  {mean_latency_ms:.2f} ms (+/- {std_latency_ms:.2f} ms)")
    print(f"  P95 Latency:         {p95_latency_ms:.2f} ms")
    print(f"  Observed FPS:        {fps:.2f} FPS")
    print(f"  Model Size:          {model_size_mb:.2f} MB")
    print(f"  Real-Time Status:    {'READY (FPS > 30)' if fps >= 30.0 else 'SUB-REALTIME'}")
    print("=" * 70)

    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 Model Latency and FPS")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")

    args = parser.parse_args()
    benchmark_model(model_path=args.model, imgsz=args.imgsz, device=args.device)
