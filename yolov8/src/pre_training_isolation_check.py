"""
Pre-Training Cryptographic Data Isolation and Leakage Verification for Model B
"""

import os
import sys
import glob
import hashlib
from collections import defaultdict

def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_isolation():
    print("=" * 80)
    print("CRITICAL PRE-TRAINING DATA ISOLATION VERIFICATION FOR MODEL B")
    print("=" * 80)

    train_imgs = glob.glob("yolov8/data/baseline/train/images/*.*") + glob.glob("yolov8/data/camo_training/train/images/*.*")
    val_imgs = glob.glob("yolov8/data/baseline/val/images/*.*") + glob.glob("yolov8/data/camo_training/val/images/*.*")
    
    baseline_test_imgs = glob.glob("yolov8/data/baseline/test/images/*.*")
    heldout_camo_imgs = glob.glob("yolov8/data/camo_benchmark/4class_camo/images/*.*")
    negative_imgs = glob.glob("yolov8/data/camo_benchmark/negatives/images/*.*")

    print(f"Model B Training Images:       {len(train_imgs):>5} (Baseline: 1036, Camo Train: 229)")
    print(f"Model B Validation Images:     {len(val_imgs):>5} (Baseline: 216, Camo Val: 45)")
    print(f"Baseline Test Images (Frozen): {len(baseline_test_imgs):>5}")
    print(f"Held-Out Camo Benchmark:       {len(heldout_camo_imgs):>5}")
    print(f"True Negative Benchmark:       {len(negative_imgs):>5}")

    train_hashes = {compute_sha256(p): p for p in train_imgs}
    val_hashes = {compute_sha256(p): p for p in val_imgs}
    baseline_test_hashes = {compute_sha256(p): p for p in baseline_test_imgs}
    heldout_camo_hashes = {compute_sha256(p): p for p in heldout_camo_imgs}

    # Verify zero overlap
    train_camo_overlap = set(train_hashes.keys()).intersection(heldout_camo_hashes.keys())
    val_camo_overlap = set(val_hashes.keys()).intersection(heldout_camo_hashes.keys())
    train_base_test_overlap = set(train_hashes.keys()).intersection(baseline_test_hashes.keys())
    val_base_test_overlap = set(val_hashes.keys()).intersection(baseline_test_hashes.keys())

    print("-" * 80)
    print(f"Overlap: Model B Train <-> Held-Out Camo Benchmark: {len(train_camo_overlap)}")
    print(f"Overlap: Model B Val   <-> Held-Out Camo Benchmark: {len(val_camo_overlap)}")
    print(f"Overlap: Model B Train <-> Baseline Test Set:      {len(train_base_test_overlap)}")
    print(f"Overlap: Model B Val   <-> Baseline Test Set:      {len(val_base_test_overlap)}")

    is_isolated = (
        len(train_camo_overlap) == 0 and
        len(val_camo_overlap) == 0 and
        len(train_base_test_overlap) == 0 and
        len(val_base_test_overlap) == 0
    )

    print("-" * 80)
    print(f"ISOLATION STATUS: {'100% DISJOINT & VERIFIED' if is_isolated else 'LEAKAGE DETECTED - ABORT'}")
    print("=" * 80)
    return is_isolated

if __name__ == "__main__":
    if not verify_isolation():
        sys.exit(1)
