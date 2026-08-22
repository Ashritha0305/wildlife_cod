"""
Stage 6A: Comprehensive Comparative Evaluation of Model A vs Model B
Evaluates both models on:
  1. Normal Baseline Test Set (211 images, 352 instances)
  2. Held-Out 4-Class Camouflage Benchmark (45 images, 64 instances)
  3. Animal-Free True Negative Benchmark (60 images)
Measures latency breakdown, real-stream FPS, FPPI, and per-class metrics on RTX 4050 GPU.
"""

import os
import sys
import glob
import time
import json
import torch
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_A_PATH = os.path.abspath("yolov8/results/baseline_yolov8s/weights/best.pt")
MODEL_B_PATH = os.path.abspath("yolov8/results/camouflage_yolov8s/weights/best.pt")

BASELINE_CFG = os.path.abspath("yolov8/config/baseline.yaml")
CAMO_CFG = os.path.abspath("yolov8/config/camo_benchmark.yaml")
NEG_CFG = os.path.abspath("yolov8/config/negatives_benchmark.yaml")

CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]


def evaluate_model_on_dataset(model: YOLO, data_cfg: str, split_name: str = "test"):
    """Run Ultralytics val mode to get official metrics."""
    metrics = model.val(
        data=data_cfg,
        split=split_name,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        save_json=False,
        plots=False,
        verbose=False
    )
    
    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map)
    }

    per_class = {}
    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    ap50_per_class = metrics.box.ap50
    ap_per_class = metrics.box.ap

    for i, cname in enumerate(CLASS_NAMES):
        p_val = float(p_per_class[i]) if i < len(p_per_class) else 0.0
        r_val = float(r_per_class[i]) if i < len(r_per_class) else 0.0
        ap50_val = float(ap50_per_class[i]) if i < len(ap50_per_class) else 0.0
        ap_val = float(ap_per_class[i]) if i < len(ap_per_class) else 0.0

        per_class[cname] = {
            "precision": p_val,
            "recall": r_val,
            "ap50": ap50_val,
            "ap50_95": ap_val
        }

    return overall, per_class


def measure_fppi_on_negatives(model: YOLO, neg_dir: str = "yolov8/data/camo_benchmark/negatives/images", conf: float = 0.25):
    """Count false positive detections on pure animal-free background images."""
    neg_images = glob.glob(os.path.join(neg_dir, "*.*"))
    total_fp = 0
    per_class_fp = {c: 0 for c in CLASS_NAMES}

    for img_p in neg_images:
        img = cv2.imread(img_p)
        if img is None:
            continue
        res = model.predict(img, imgsz=640, conf=conf, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
        boxes = res[0].boxes
        if boxes is not None and len(boxes) > 0:
            total_fp += len(boxes)
            for cls_idx in boxes.cls.cpu().numpy():
                c_int = int(cls_idx)
                if c_int < len(CLASS_NAMES):
                    per_class_fp[CLASS_NAMES[c_int]] += 1

    fppi = total_fp / max(1, len(neg_images))
    return len(neg_images), total_fp, fppi, per_class_fp


def benchmark_hardware_latency(model_path: str, test_images_dir: str = "yolov8/data/baseline/test/images", iterations: int = 150):
    """Measure exact preprocessing, inference, and postprocessing latencies on RTX 4050."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)

    test_imgs = glob.glob(os.path.join(test_images_dir, "*.*"))[:iterations]
    if len(test_imgs) < iterations:
        test_imgs = (test_imgs * (iterations // len(test_imgs) + 1))[:iterations]

    loaded_frames = [cv2.imread(p) for p in test_imgs]

    # Warmup
    print("  Warming up GPU with 25 iterations...")
    for i in range(25):
        _ = model.predict(loaded_frames[i % len(loaded_frames)], imgsz=640, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"  Benchmarking across {len(loaded_frames)} test stream frames...")
    latencies = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    for frame in loaded_frames:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        res = model.predict(frame, imgsz=640, device=0, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_total_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / avg_total_lat
    
    # Internal breakdown from Ultralytics speed dict of last run
    speed_dict = res[0].speed
    prep_lat = speed_dict.get("preprocess", 0.0)
    inf_lat = speed_dict.get("inference", 0.0)
    post_lat = speed_dict.get("postprocess", 0.0)
    
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0

    return {
        "preprocess_ms": prep_lat,
        "inference_ms": inf_lat,
        "postprocess_ms": post_lat,
        "total_latency_ms": avg_total_lat,
        "std_latency_ms": std_lat,
        "fps": fps,
        "peak_vram_mb": peak_vram
    }


def run_comparative_evaluation():
    print("=" * 80)
    print("STAGE 6A: SCIENTIFIC COMPARATIVE EVALUATION — MODEL A vs MODEL B")
    print("=" * 80)

    print(f"Model A (Baseline):   {MODEL_A_PATH}")
    print(f"Model B (Camouflage): {MODEL_B_PATH}")

    model_a = YOLO(MODEL_A_PATH)
    model_b = YOLO(MODEL_B_PATH)

    # 1. Evaluate on Normal Baseline Test Set
    print("\n--- 1. EVALUATION ON NORMAL BASELINE TEST SET (211 images, 352 instances) ---")
    ov_a_norm, pc_a_norm = evaluate_model_on_dataset(model_a, BASELINE_CFG, split_name="test")
    ov_b_norm, pc_b_norm = evaluate_model_on_dataset(model_b, BASELINE_CFG, split_name="test")

    print(f"Model A (Normal Test): P={ov_a_norm['precision']*100:.2f}%, R={ov_a_norm['recall']*100:.2f}%, mAP50={ov_a_norm['map50']*100:.2f}%, mAP50-95={ov_a_norm['map50_95']*100:.2f}%")
    print(f"Model B (Normal Test): P={ov_b_norm['precision']*100:.2f}%, R={ov_b_norm['recall']*100:.2f}%, mAP50={ov_b_norm['map50']*100:.2f}%, mAP50-95={ov_b_norm['map50_95']*100:.2f}%")

    # 2. Evaluate on Held-Out Camouflage Benchmark
    print("\n--- 2. EVALUATION ON HELD-OUT CAMOUFLAGE BENCHMARK (45 images, 64 instances) ---")
    ov_a_camo, pc_a_camo = evaluate_model_on_dataset(model_a, CAMO_CFG, split_name="test")
    ov_b_camo, pc_b_camo = evaluate_model_on_dataset(model_b, CAMO_CFG, split_name="test")

    print(f"Model A (Camo Test):   P={ov_a_camo['precision']*100:.2f}%, R={ov_a_camo['recall']*100:.2f}%, mAP50={ov_a_camo['map50']*100:.2f}%, mAP50-95={ov_a_camo['map50_95']*100:.2f}%")
    print(f"Model B (Camo Test):   P={ov_b_camo['precision']*100:.2f}%, R={ov_b_camo['recall']*100:.2f}%, mAP50={ov_b_camo['map50']*100:.2f}%, mAP50-95={ov_b_camo['map50_95']*100:.2f}%")

    # 3. Evaluate on True Negative Benchmark
    print("\n--- 3. EVALUATION ON TRUE NEGATIVE BACKGROUNDS (60 images) ---")
    n_a, fp_a, fppi_a, pc_fp_a = measure_fppi_on_negatives(model_a)
    n_b, fp_b, fppi_b, pc_fp_b = measure_fppi_on_negatives(model_b)

    print(f"Model A: False Positives = {fp_a} across {n_a} images -> FPPI = {fppi_a:.4f} (By Class: {pc_fp_a})")
    print(f"Model B: False Positives = {fp_b} across {n_b} images -> FPPI = {fppi_b:.4f} (By Class: {pc_fp_b})")

    # 4. Hardware Benchmarking on RTX 4050
    print("\n--- 4. HARDWARE LATENCY & THROUGHPUT BENCHMARK (RTX 4050 GPU) ---")
    bench_b = benchmark_hardware_latency(MODEL_B_PATH)
    print(f"Model B Real-Stream Latency: {bench_b['total_latency_ms']:.2f} ms (±{bench_b['std_latency_ms']:.2f} ms) -> {bench_b['fps']:.2f} FPS")
    print(f"  Preprocess: {bench_b['preprocess_ms']:.2f} ms, Inference: {bench_b['inference_ms']:.2f} ms, Postprocess: {bench_b['postprocess_ms']:.2f} ms")
    print(f"  Peak VRAM:  {bench_b['peak_vram_mb']:.2f} MB")

    # Save complete evaluation JSON
    eval_report = {
        "model_a": {
            "path": MODEL_A_PATH,
            "normal_test": {"overall": ov_a_norm, "per_class": pc_a_norm},
            "camo_test": {"overall": ov_a_camo, "per_class": pc_a_camo},
            "negatives": {"total_images": n_a, "false_positives": fp_a, "fppi": fppi_a, "per_class_fp": pc_fp_a}
        },
        "model_b": {
            "path": MODEL_B_PATH,
            "normal_test": {"overall": ov_b_norm, "per_class": pc_b_norm},
            "camo_test": {"overall": ov_b_camo, "per_class": pc_b_camo},
            "negatives": {"total_images": n_b, "false_positives": fp_b, "fppi": fppi_b, "per_class_fp": pc_fp_b},
            "benchmark": bench_b
        }
    }

    out_json = "yolov8/results/camouflage_yolov8s/comparative_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nSaved complete evaluation metrics to {out_json}")


if __name__ == "__main__":
    run_comparative_evaluation()
