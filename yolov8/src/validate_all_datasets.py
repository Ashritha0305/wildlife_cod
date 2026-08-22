"""
Comprehensive Global Dataset Validation and Cryptographic Leakage Audit
Audits Baseline, Camouflage Training, Held-Out Camouflage Benchmark, and True Negatives.
"""

import os
import sys
import glob
import hashlib
from collections import defaultdict
import cv2

sys.path.insert(0, os.path.abspath("."))
from yolov8.src.validate_dataset import DatasetValidator

CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]


def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_global_audit():
    print("=" * 80)
    print("COMPREHENSIVE GLOBAL DATASET INTEGRITY & CRYPTOGRAPHIC LEAKAGE AUDIT")
    print("=" * 80)

    # 1. Audit Baseline
    print("\n--- 1. AUDITING BASELINE DATASET (yolov8/data/baseline) ---")
    val_baseline = DatasetValidator("yolov8/data/baseline", CLASS_NAMES)
    val_baseline.run_audit()
    print(val_baseline.generate_report())

    # 2. Audit Camo Training
    print("\n--- 2. AUDITING CAMOUFLAGE TRAINING SUITE (yolov8/data/camo_training) ---")
    val_camo_train = DatasetValidator("yolov8/data/camo_training", CLASS_NAMES)
    val_camo_train.run_audit()
    print(val_camo_train.generate_report())

    # 3. Audit Camo Benchmark (4class_camo)
    print("\n--- 3. AUDITING HELD-OUT CAMO BENCHMARK (yolov8/data/camo_benchmark/4class_camo) ---")
    # For single directory benchmark, check files directly
    camo_test_imgs = sorted(glob.glob("yolov8/data/camo_benchmark/4class_camo/images/*.*"))
    camo_test_lbls = sorted(glob.glob("yolov8/data/camo_benchmark/4class_camo/labels/*.txt"))
    print(f"Held-Out Camo Benchmark Images: {len(camo_test_imgs)}")
    print(f"Held-Out Camo Benchmark Labels: {len(camo_test_lbls)}")

    camo_test_classes = defaultdict(int)
    camo_test_instances = 0
    camo_test_errors = []

    for img_p in camo_test_imgs:
        base = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join("yolov8/data/camo_benchmark/4class_camo/labels", f"{base}.txt")
        if not os.path.exists(lbl_p):
            camo_test_errors.append(f"Missing label for {img_p}")
            continue

        img = cv2.imread(img_p)
        if img is None:
            camo_test_errors.append(f"Corrupt image {img_p}")
            continue

        with open(lbl_p, "r", encoding="utf-8") as lf:
            lines = [l.strip() for l in lf if l.strip()]

        for l in lines:
            parts = l.split()
            if len(parts) != 5:
                camo_test_errors.append(f"Invalid tokens in {lbl_p}: {l}")
                continue
            cid = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                camo_test_errors.append(f"Out of bounds box in {lbl_p}: {l}")
                continue
            camo_test_classes[CLASS_NAMES[cid]] += 1
            camo_test_instances += 1

    print(f"Held-Out Camo Instances: {camo_test_instances}")
    for cname in CLASS_NAMES:
        print(f"  {cname:<12}: {camo_test_classes[cname]} instances")
    print(f"Errors detected: {len(camo_test_errors)}")

    # 4. Audit Negatives
    print("\n--- 4. AUDITING TRUE NEGATIVES (yolov8/data/camo_benchmark/negatives) ---")
    neg_imgs = sorted(glob.glob("yolov8/data/camo_benchmark/negatives/images/*.*"))
    neg_lbls = sorted(glob.glob("yolov8/data/camo_benchmark/negatives/labels/*.txt"))
    print(f"Negative Images: {len(neg_imgs)}, Negative Labels: {len(neg_lbls)}")
    neg_non_empty = 0
    for lp in neg_lbls:
        with open(lp, "r") as f:
            if f.read().strip():
                neg_non_empty += 1
    print(f"Non-empty negative labels: {neg_non_empty} (should be 0)")

    # 5. Global Cryptographic SHA-256 Cross-Partition Leakage Audit
    print("\n" + "=" * 80)
    print("GLOBAL SHA-256 CROSS-PARTITION DEDUPLICATION & LEAKAGE AUDIT")
    print("=" * 80)

    partitions = {
        "baseline/train": glob.glob("yolov8/data/baseline/train/images/*.*"),
        "baseline/val": glob.glob("yolov8/data/baseline/val/images/*.*"),
        "baseline/test": glob.glob("yolov8/data/baseline/test/images/*.*"),
        "camo_training/train": glob.glob("yolov8/data/camo_training/train/images/*.*"),
        "camo_training/val": glob.glob("yolov8/data/camo_training/val/images/*.*"),
        "camo_benchmark/4class_camo": glob.glob("yolov8/data/camo_benchmark/4class_camo/images/*.*"),
        "camo_benchmark/negatives": glob.glob("yolov8/data/camo_benchmark/negatives/images/*.*")
    }

    global_hash_map = defaultdict(list)
    total_files = 0

    for part_name, file_list in partitions.items():
        print(f"  Hashing {part_name:<30}: {len(file_list):>5} images")
        total_files += len(file_list)
        for fpath in file_list:
            h = compute_sha256(fpath)
            global_hash_map[h].append((part_name, fpath))

    cross_leakage = []
    intra_duplicates = []

    for h, entries in global_hash_map.items():
        if len(entries) > 1:
            part_set = {e[0] for e in entries}
            if len(part_set) > 1:
                cross_leakage.append((h, entries))
            else:
                intra_duplicates.append((h, entries))

    print("-" * 80)
    print(f"Total Images Hashed: {total_files}")
    print(f"Unique SHA-256 Hashes: {len(global_hash_map)}")
    print(f"Intra-Split Duplicate Pairs: {len(intra_duplicates)}")
    print(f"Cross-Partition Leakage Collisions: {len(cross_leakage)}")
    if cross_leakage:
        print("CRITICAL LEAKAGE DETECTED:")
        for h, entries in cross_leakage:
            print(f"  Hash {h[:12]} in {entries}")
    else:
        print("ZERO LEAKAGE CONFIRMED: All partitions are 100% cryptographically disjoint.")
    print("=" * 80)


if __name__ == "__main__":
    run_global_audit()
