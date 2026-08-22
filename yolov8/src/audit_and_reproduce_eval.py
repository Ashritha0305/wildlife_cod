"""
Stage 6B: Comprehensive Reproducibility and Metric Audit Engine
Performs standardized, fully-reproducible evaluation of Model A vs Model B:
  1. Cryptographic and parameter audit of Checkpoint A & Checkpoint B
  2. Dataset instance and hash audit
  3. Re-evaluation of Model A and Model B under both:
     a) Production Threshold (conf=0.25, iou=0.6) - matches Stage 4A-2 / validate.py
     b) Standard Benchmark Threshold (conf=0.001, iou=0.7) - standard COCO protocol
  4. Isolation of results in:
     - yolov8/results/repro_eval_modelA/
     - yolov8/results/repro_eval_modelB_normal/
     - yolov8/results/repro_eval_modelA_camo/
     - yolov8/results/repro_eval_modelB_camo/
  5. Negative benchmark FPPI evaluation
  6. Latency, FPS, and VRAM benchmarking with CUDA sync
"""

import os
import sys
import glob
import json
import time
import hashlib
import platform
import shutil
import numpy as np
import cv2
import torch
from ultralytics import YOLO

MODEL_A_PATH = os.path.abspath("yolov8/results/baseline_yolov8s/weights/best.pt")
MODEL_B_PATH = os.path.abspath("yolov8/results/camouflage_yolov8s/weights/best.pt")

BASELINE_CFG = os.path.abspath("yolov8/config/baseline.yaml")
CAMO_CFG = os.path.abspath("yolov8/config/camo_benchmark.yaml")
NEG_CFG = os.path.abspath("yolov8/config/negatives_benchmark.yaml")

DIR_REPRO_A_NORM = os.path.abspath("yolov8/results/repro_eval_modelA")
DIR_REPRO_B_NORM = os.path.abspath("yolov8/results/repro_eval_modelB_normal")
DIR_REPRO_A_CAMO = os.path.abspath("yolov8/results/repro_eval_modelA_camo")
DIR_REPRO_B_CAMO = os.path.abspath("yolov8/results/repro_eval_modelB_camo")

CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]


def get_file_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def audit_checkpoints():
    print("\n" + "=" * 80)
    print("STEP 1: CHECKPOINT VERIFICATION & INTEGRITY AUDIT")
    print("=" * 80)

    for name, path in [("Model A (Baseline)", MODEL_A_PATH), ("Model B (Camouflage)", MODEL_B_PATH)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        size_bytes = os.path.getsize(path)
        sha = get_file_sha256(path)
        
        # Load weights metadata
        model = YOLO(path)
        num_params = sum(p.numel() for p in model.model.parameters())
        classes = model.names

        print(f"[{name}]")
        print(f"  Path:          {path}")
        print(f"  File Size:     {size_bytes} bytes ({size_bytes / (1024**2):.2f} MB)")
        print(f"  SHA-256 Hash:  {sha}")
        print(f"  Architecture:  YOLOv8s ({num_params:,} parameters)")
        print(f"  Class Names:   {classes}")


def audit_test_dataset(test_img_dir="yolov8/data/baseline/test/images", test_lbl_dir="yolov8/data/baseline/test/labels"):
    print("\n" + "=" * 80)
    print("STEP 2: TEST DATASET VERIFICATION")
    print("=" * 80)

    images = glob.glob(os.path.join(test_img_dir, "*.*"))
    labels = glob.glob(os.path.join(test_lbl_dir, "*.txt"))

    total_instances = 0
    per_class_counts = {i: 0 for i in range(len(CLASS_NAMES))}

    for lbl in labels:
        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    total_instances += 1
                    per_class_counts[cls_id] = per_class_counts.get(cls_id, 0) + 1

    print(f"Baseline Test Directory: {test_img_dir}")
    print(f"  Image Count:           {len(images)} images")
    print(f"  Label Count:           {len(labels)} label files")
    print(f"  Total GT Instances:    {total_instances} instances")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"    - {cname:<10}: {per_class_counts.get(i, 0)} instances")

    return len(images), total_instances, per_class_counts


def audit_software_environment():
    print("\n" + "=" * 80)
    print("STEP 4: SOFTWARE & HARDWARE ENVIRONMENT AUDIT")
    print("=" * 80)

    import ultralytics
    import numpy

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"

    env_info = {
        "os": platform.system() + " " + platform.release(),
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics.__version__,
        "pytorch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "cuda_version": cuda_ver,
        "gpu": gpu_name,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

    for k, v in env_info.items():
        print(f"  {k:<22}: {v}")

    return env_info


def run_formal_evaluation(model_path: str, data_yaml: str, split_name: str, save_dir: str, conf: float = 0.25, iou: float = 0.6):
    """Run Ultralytics val mode, copy plots to target directory, and return structured metrics."""
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        split=split_name,
        imgsz=640,
        batch=16,
        conf=conf,
        iou=iou,
        device=0 if torch.cuda.is_available() else "cpu",
        project=os.path.dirname(save_dir),
        name=os.path.basename(save_dir),
        exist_ok=True,
        plots=True,
        verbose=False
    )

    box = metrics.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map)
    }

    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        p_val = float(box.p[i]) if i < len(box.p) else 0.0
        r_val = float(box.r[i]) if i < len(box.r) else 0.0
        ap50_val = float(box.ap50[i]) if i < len(box.ap50) else 0.0
        ap_val = float(box.ap[i]) if i < len(box.ap) else 0.0
        per_class[cname] = {
            "precision": p_val,
            "recall": r_val,
            "ap50": ap50_val,
            "ap50_95": ap_val
        }

    return overall, per_class, metrics.speed


def evaluate_negatives(model_path: str, neg_img_dir: str = "yolov8/data/camo_benchmark/negatives/images", conf: float = 0.25):
    model = YOLO(model_path)
    images = sorted(glob.glob(os.path.join(neg_img_dir, "*.*")))
    total_fps = 0
    per_class_fps = {c: 0 for c in CLASS_NAMES}

    for img_p in images:
        img = cv2.imread(img_p)
        if img is None:
            continue
        res = model.predict(img, imgsz=640, conf=conf, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
        boxes = res[0].boxes
        if boxes is not None and len(boxes) > 0:
            total_fps += len(boxes)
            for cls_idx in boxes.cls.cpu().numpy():
                c_int = int(cls_idx)
                if c_int < len(CLASS_NAMES):
                    per_class_fps[CLASS_NAMES[c_int]] += 1

    fppi = total_fps / max(1, len(images))
    return len(images), total_fps, fppi, per_class_fps


def benchmark_model_speed(model_path: str, test_images_dir: str = "yolov8/data/baseline/test/images", iterations: int = 150):
    model = YOLO(model_path)
    test_imgs = glob.glob(os.path.join(test_images_dir, "*.*"))[:iterations]
    if len(test_imgs) < iterations:
        test_imgs = (test_imgs * (iterations // len(test_imgs) + 1))[:iterations]
    frames = [cv2.imread(p) for p in test_imgs]

    # Warmup
    for i in range(25):
        _ = model.predict(frames[i % len(frames)], imgsz=640, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

    latencies = []
    for frame in frames:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=640, device=0, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / avg_lat
    speed_dict = res[0].speed
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0

    return {
        "preprocess_ms": speed_dict.get("preprocess", 0.0),
        "inference_ms": speed_dict.get("inference", 0.0),
        "postprocess_ms": speed_dict.get("postprocess", 0.0),
        "total_latency_ms": avg_lat,
        "std_latency_ms": std_lat,
        "fps": fps,
        "peak_vram_mb": peak_vram
    }


def execute_stage6b_audit():
    print("=" * 80)
    print("STAGE 6B: FULL REPRODUCIBILITY, ISOLATION & METRIC AUDIT")
    print("=" * 80)

    # 1. Audit Checkpoints
    audit_checkpoints()

    # 2. Audit Datasets
    n_imgs, n_inst, pc_inst = audit_test_dataset()

    # 3. Audit Environment
    env = audit_software_environment()

    # 4. Standard Re-Evaluation (Production threshold: conf=0.25, iou=0.6)
    print("\n" + "=" * 80)
    print("STEP 5 & 6: RE-EVALUATION ON NORMAL BASELINE TEST SET (conf=0.25, iou=0.6)")
    print("=" * 80)

    ov_a_prod, pc_a_prod, speed_a = run_formal_evaluation(MODEL_A_PATH, BASELINE_CFG, "test", DIR_REPRO_A_NORM, conf=0.25, iou=0.6)
    ov_b_prod, pc_b_prod, speed_b = run_formal_evaluation(MODEL_B_PATH, BASELINE_CFG, "test", DIR_REPRO_B_NORM, conf=0.25, iou=0.6)

    print(f"Model A (Normal Test, conf=0.25): P={ov_a_prod['precision']*100:.2f}%, R={ov_a_prod['recall']*100:.2f}%, mAP50={ov_a_prod['map50']*100:.2f}%, mAP50-95={ov_a_prod['map50_95']*100:.2f}%")
    print(f"Model B (Normal Test, conf=0.25): P={ov_b_prod['precision']*100:.2f}%, R={ov_b_prod['recall']*100:.2f}%, mAP50={ov_b_prod['map50']*100:.2f}%, mAP50-95={ov_b_prod['map50_95']*100:.2f}%")

    # 5. Full-Recall Benchmark Re-Evaluation (conf=0.001, iou=0.7)
    print("\n--- Standard COCO Protocol Evaluation (conf=0.001, iou=0.7) ---")
    ov_a_coco, pc_a_coco, _ = run_formal_evaluation(MODEL_A_PATH, BASELINE_CFG, "test", os.path.join(DIR_REPRO_A_NORM, "coco_conf0001"), conf=0.001, iou=0.7)
    ov_b_coco, pc_b_coco, _ = run_formal_evaluation(MODEL_B_PATH, BASELINE_CFG, "test", os.path.join(DIR_REPRO_B_NORM, "coco_conf0001"), conf=0.001, iou=0.7)
    print(f"Model A (Normal Test, conf=0.001): P={ov_a_coco['precision']*100:.2f}%, R={ov_a_coco['recall']*100:.2f}%, mAP50={ov_a_coco['map50']*100:.2f}%, mAP50-95={ov_a_coco['map50_95']*100:.2f}%")
    print(f"Model B (Normal Test, conf=0.001): P={ov_b_coco['precision']*100:.2f}%, R={ov_b_coco['recall']*100:.2f}%, mAP50={ov_b_coco['map50']*100:.2f}%, mAP50-95={ov_b_coco['map50_95']*100:.2f}%")

    # 6. Camouflage Re-Evaluation (Production threshold: conf=0.25, iou=0.6)
    print("\n" + "=" * 80)
    print("STEP 7: RE-EVALUATION ON HELD-OUT CAMOUFLAGE BENCHMARK (conf=0.25, iou=0.6)")
    print("=" * 80)
    ov_a_camo_prod, pc_a_camo_prod, _ = run_formal_evaluation(MODEL_A_PATH, CAMO_CFG, "test", DIR_REPRO_A_CAMO, conf=0.25, iou=0.6)
    ov_b_camo_prod, pc_b_camo_prod, _ = run_formal_evaluation(MODEL_B_PATH, CAMO_CFG, "test", DIR_REPRO_B_CAMO, conf=0.25, iou=0.6)

    print(f"Model A (Camo Test, conf=0.25): P={ov_a_camo_prod['precision']*100:.2f}%, R={ov_a_camo_prod['recall']*100:.2f}%, mAP50={ov_a_camo_prod['map50']*100:.2f}%, mAP50-95={ov_a_camo_prod['map50_95']*100:.2f}%")
    print(f"Model B (Camo Test, conf=0.25): P={ov_b_camo_prod['precision']*100:.2f}%, R={ov_b_camo_prod['recall']*100:.2f}%, mAP50={ov_b_camo_prod['map50']*100:.2f}%, mAP50-95={ov_b_camo_prod['map50_95']*100:.2f}%")

    # 7. Camouflage Re-Evaluation (Standard COCO protocol: conf=0.001, iou=0.7)
    ov_a_camo_coco, pc_a_camo_coco, _ = run_formal_evaluation(MODEL_A_PATH, CAMO_CFG, "test", os.path.join(DIR_REPRO_A_CAMO, "coco_conf0001"), conf=0.001, iou=0.7)
    ov_b_camo_coco, pc_b_camo_coco, _ = run_formal_evaluation(MODEL_B_PATH, CAMO_CFG, "test", os.path.join(DIR_REPRO_B_CAMO, "coco_conf0001"), conf=0.001, iou=0.7)

    # 8. Negative Evaluation (conf=0.25)
    print("\n" + "=" * 80)
    print("STEP 8: RE-EVALUATION ON NEGATIVE BACKGROUNDS (60 images, conf=0.25)")
    print("=" * 80)
    n_a, fp_a, fppi_a, pcf_a = evaluate_negatives(MODEL_A_PATH, conf=0.25)
    n_b, fp_b, fppi_b, pcf_b = evaluate_negatives(MODEL_B_PATH, conf=0.25)
    print(f"Model A: FPs={fp_a}, FPPI={fppi_a:.4f}, by class={pcf_a}")
    print(f"Model B: FPs={fp_b}, FPPI={fppi_b:.4f}, by class={pcf_b}")

    # 9. Hardware Benchmark
    print("\n" + "=" * 80)
    print("STEP 9: HARDWARE BENCHMARKS ON RTX 4050 GPU")
    print("=" * 80)
    bench_a = benchmark_model_speed(MODEL_A_PATH)
    bench_b = benchmark_model_speed(MODEL_B_PATH)

    print(f"Model A: {bench_a['total_latency_ms']:.2f} ms (±{bench_a['std_latency_ms']:.2f} ms) -> {bench_a['fps']:.2f} FPS | Peak VRAM: {bench_a['peak_vram_mb']:.2f} MB")
    print(f"Model B: {bench_b['total_latency_ms']:.2f} ms (±{bench_b['std_latency_ms']:.2f} ms) -> {bench_b['fps']:.2f} FPS | Peak VRAM: {bench_b['peak_vram_mb']:.2f} MB")

    # Save comprehensive audit record
    audit_data = {
        "environment": env,
        "checkpoints": {
            "model_a": {
                "path": MODEL_A_PATH,
                "sha256": get_file_sha256(MODEL_A_PATH),
                "size_bytes": os.path.getsize(MODEL_A_PATH)
            },
            "model_b": {
                "path": MODEL_B_PATH,
                "sha256": get_file_sha256(MODEL_B_PATH),
                "size_bytes": os.path.getsize(MODEL_B_PATH)
            }
        },
        "dataset_baseline_test": {
            "images": n_imgs,
            "instances": n_inst,
            "per_class_instances": pc_inst
        },
        "production_eval_conf025": {
            "model_a": {
                "normal_test": {"overall": ov_a_prod, "per_class": pc_a_prod},
                "camo_test": {"overall": ov_a_camo_prod, "per_class": pc_a_camo_prod},
                "negatives": {"total_images": n_a, "false_positives": fp_a, "fppi": fppi_a, "per_class_fp": pcf_a},
                "benchmark": bench_a
            },
            "model_b": {
                "normal_test": {"overall": ov_b_prod, "per_class": pc_b_prod},
                "camo_test": {"overall": ov_b_camo_prod, "per_class": pc_b_camo_prod},
                "negatives": {"total_images": n_b, "false_positives": fp_b, "fppi": fppi_b, "per_class_fp": pcf_b},
                "benchmark": bench_b
            }
        },
        "standard_coco_eval_conf0001": {
            "model_a": {
                "normal_test": {"overall": ov_a_coco, "per_class": pc_a_coco},
                "camo_test": {"overall": ov_a_camo_coco, "per_class": pc_a_camo_coco}
            },
            "model_b": {
                "normal_test": {"overall": ov_b_coco, "per_class": pc_b_coco},
                "camo_test": {"overall": ov_b_camo_coco, "per_class": pc_b_camo_coco}
            }
        }
    }

    out_file = "yolov8/results/stage6b_audit_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAudit complete! Saved comprehensive audit record to {out_file}")


if __name__ == "__main__":
    execute_stage6b_audit()
