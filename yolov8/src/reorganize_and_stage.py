"""
Dataset Organization & Camouflage Staging Script for YOLOv8 Wildlife Subsystem
Preserves the verified Baseline dataset, stages the Camouflage Benchmark,
and prepares the Multi-Source dataset structure.
"""

import os
import sys
import shutil
import glob
import hashlib
from typing import Dict, List, Set, Tuple


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def preserve_baseline(data_root: str):
    """Copy verified 4-class dataset to yolov8/data/baseline/."""
    baseline_root = os.path.join(data_root, "baseline")
    print(f"Preserving baseline dataset into: {baseline_root}")
    
    for split in ["train", "val", "test"]:
        src_img = os.path.join(data_root, split, "images")
        src_lbl = os.path.join(data_root, split, "labels")
        dst_img = os.path.join(baseline_root, split, "images")
        dst_lbl = os.path.join(baseline_root, split, "labels")
        
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        
        if os.path.exists(src_img):
            for f in os.listdir(src_img):
                shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))
        if os.path.exists(src_lbl):
            for f in os.listdir(src_lbl):
                shutil.copy2(os.path.join(src_lbl, f), os.path.join(dst_lbl, f))
                
    print("Baseline preservation complete.")


if __name__ == "__main__":
    data_dir = "yolov8/data"
    preserve_baseline(data_dir)
