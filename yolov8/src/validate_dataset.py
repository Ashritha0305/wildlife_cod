"""
Comprehensive Dataset Integrity Validator for YOLOv8 Wildlife Detection Subsystem
Audits directory structure, image readability, label syntax, bounding-box coordinate
bounds, class frequencies, negative background samples, and cross-split leakage.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import os
import sys
import glob
import hashlib
from collections import defaultdict
import cv2
import numpy as np


class DatasetValidator:
    def __init__(self, data_root: str, class_names: Optional[List[str]] = None):
        self.data_root = os.path.abspath(data_root)
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        
        self.splits = ["train", "val", "test"]
        self.stats = {
            "total_images": 0,
            "split_images": defaultdict(int),
            "split_labels": defaultdict(int),
            "total_instances": 0,
            "class_distribution": defaultdict(lambda: defaultdict(int)),
            "negative_images": defaultdict(int),
            "corrupted_images": [],
            "missing_labels": [],
            "orphan_labels": [],
            "invalid_labels": [],
            "out_of_bounds_boxes": [],
            "zero_area_boxes": [],
            "duplicate_hashes": defaultdict(list),
            "cross_split_leakage": []
        }
        
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file to detect duplicates and data leakage."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_split(self, split_name: str) -> Dict[str, Any]:
        """Validate a single partition (train/val/test)."""
        split_dir = os.path.join(self.data_root, split_name)
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        
        if not os.path.isdir(img_dir):
            return {"error": f"Image directory missing: {img_dir}"}
        if not os.path.isdir(lbl_dir):
            return {"error": f"Label directory missing: {lbl_dir}"}
            
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        img_files_set = set()
        for ext in image_extensions:
            for p in glob.glob(os.path.join(img_dir, ext)):
                img_files_set.add(os.path.normpath(p))
            for p in glob.glob(os.path.join(img_dir, ext.upper())):
                img_files_set.add(os.path.normpath(p))
        img_files = sorted(list(img_files_set))
            
        lbl_files = sorted(list({os.path.normpath(p) for p in glob.glob(os.path.join(lbl_dir, "*.txt"))}))
        
        lbl_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in lbl_files}
        img_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
        
        # Check orphan labels (labels without images)
        for base, lbl_path in lbl_basenames.items():
            if base not in img_basenames:
                self.stats["orphan_labels"].append(lbl_path)
                
        # Validate each image and its corresponding label
        for base, img_path in img_basenames.items():
            self.stats["total_images"] += 1
            self.stats["split_images"][split_name] += 1
            
            # 1. Image readability check
            try:
                img = cv2.imread(img_path)
                if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
                    self.stats["corrupted_images"].append((img_path, "Unreadable or zero-dimension image"))
                    continue
                img_h, img_w = img.shape[:2]
            except Exception as e:
                self.stats["corrupted_images"].append((img_path, str(e)))
                continue
                
            # 2. Compute hash for duplicate and leakage audit
            img_hash = self._compute_file_hash(img_path)
            self.stats["duplicate_hashes"][img_hash].append((split_name, img_path))
            
            # 3. Label matching
            if base not in lbl_basenames:
                self.stats["missing_labels"].append(img_path)
                continue
                
            lbl_path = lbl_basenames[base]
            self.stats["split_labels"][split_name] += 1
            
            # 4. Label contents parsing
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
            except Exception as e:
                self.stats["invalid_labels"].append((lbl_path, f"Read error: {e}"))
                continue
                
            if len(lines) == 0:
                # True negative background image
                self.stats["negative_images"][split_name] += 1
                continue
                
            for line_idx, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    self.stats["invalid_labels"].append((lbl_path, f"Line {line_idx}: Expected 5 tokens, got {len(parts)}"))
                    continue
                    
                try:
                    cls_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    self.stats["invalid_labels"].append((lbl_path, f"Line {line_idx}: Non-numeric tokens in '{line}'"))
                    continue
                    
                # Validate Class ID range
                if self.num_classes > 0 and (cls_id < 0 or cls_id >= self.num_classes):
                    self.stats["invalid_labels"].append((lbl_path, f"Line {line_idx}: class_id {cls_id} out of bounds [0, {self.num_classes - 1}]"))
                    continue
                    
                # Validate normalized ranges
                if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    self.stats["out_of_bounds_boxes"].append((lbl_path, f"Line {line_idx}: Normalized coordinates outside [0.0, 1.0] in '{line}'"))
                    continue
                    
                # Validate zero-area
                if w <= 0.0 or h <= 0.0:
                    self.stats["zero_area_boxes"].append((lbl_path, f"Line {line_idx}: Non-positive dimension (w={w}, h={h})"))
                    continue
                    
                # Validate box does not exceed frame limits
                x1 = x_c - (w / 2.0)
                y1 = y_c - (h / 2.0)
                x2 = x_c + (w / 2.0)
                y2 = y_c + (h / 2.0)
                
                # Tolerant epsilon for floating-point rounding
                eps = 1e-4
                if x1 < -eps or y1 < -eps or x2 > (1.0 + eps) or y2 > (1.0 + eps):
                    self.stats["out_of_bounds_boxes"].append((lbl_path, f"Line {line_idx}: Box extends outside image frame boundaries"))
                    continue
                    
                # Record valid instance
                self.stats["total_instances"] += 1
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                self.stats["class_distribution"][cls_name][split_name] += 1
                self.stats["class_distribution"][cls_name]["total"] += 1
                
        return {"status": "success", "images": len(img_files), "labels": len(lbl_files)}

    def check_leakage(self):
        """Audit duplicate image hashes across different dataset splits."""
        for img_hash, occurrences in self.stats["duplicate_hashes"].items():
            if len(occurrences) > 1:
                splits_involved = {occ[0] for occ in occurrences}
                if len(splits_involved) > 1:
                    self.stats["cross_split_leakage"].append((img_hash, occurrences))

    def run_audit(self) -> Dict[str, Any]:
        """Execute complete audit across all dataset partitions."""
        print(f"Starting Dataset Audit on: {self.data_root}")
        for split in self.splits:
            split_path = os.path.join(self.data_root, split)
            if os.path.exists(split_path):
                res = self.validate_split(split)
                print(f"[{split.upper()}] Images: {res.get('images', 0)}, Labels: {res.get('labels', 0)}")
            else:
                print(f"[{split.upper()}] Directory not found: {split_path}")
                
        self.check_leakage()
        return self.stats

    def generate_report(self) -> str:
        """Format a clear technical report summarizing dataset audit findings."""
        report = []
        report.append("=" * 70)
        report.append("DATASET AUDIT REPORT")
        report.append("=" * 70)
        report.append(f"Root Directory: {self.data_root}")
        report.append(f"Configured Classes ({self.num_classes}): {self.class_names}")
        report.append("-" * 70)
        report.append(f"Total Images: {self.stats['total_images']}")
        report.append(f"  Train Images: {self.stats['split_images']['train']}")
        report.append(f"  Val Images:   {self.stats['split_images']['val']}")
        report.append(f"  Test Images:  {self.stats['split_images']['test']}")
        report.append(f"Total Instances: {self.stats['total_instances']}")
        report.append(f"True Negative Background Images: {sum(self.stats['negative_images'].values())}")
        for s in self.splits:
            report.append(f"  {s.capitalize()} Negatives: {self.stats['negative_images'][s]}")
        report.append("-" * 70)
        report.append("PER-CLASS INSTANCE DISTRIBUTION:")
        report.append(f"{'ID':<4} {'Species':<16} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'Share %':<8}")
        report.append("-" * 70)
        
        tot_inst = max(1, self.stats["total_instances"])
        for idx, name in enumerate(self.class_names):
            dist = self.stats["class_distribution"][name]
            train_cnt = dist["train"]
            val_cnt = dist["val"]
            test_cnt = dist["test"]
            total_cnt = dist["total"]
            share = (total_cnt / tot_inst) * 100.0
            report.append(f"{idx:<4} {name:<16} {train_cnt:<8} {val_cnt:<8} {test_cnt:<8} {total_cnt:<8} {share:>6.2f}%")
            
        report.append("-" * 70)
        report.append("DATA INTEGRITY & QUALITY CHECKS:")
        report.append(f"  Corrupted / Unreadable Images: {len(self.stats['corrupted_images'])}")
        report.append(f"  Missing Label Files:           {len(self.stats['missing_labels'])}")
        report.append(f"  Orphan Label Files:            {len(self.stats['orphan_labels'])}")
        report.append(f"  Invalid Label Lines:           {len(self.stats['invalid_labels'])}")
        report.append(f"  Out-of-Bounds Bounding Boxes:  {len(self.stats['out_of_bounds_boxes'])}")
        report.append(f"  Zero-Area Bounding Boxes:      {len(self.stats['zero_area_boxes'])}")
        report.append(f"  Cross-Split Data Leakage:      {len(self.stats['cross_split_leakage'])}")
        
        is_pass = (
            len(self.stats['corrupted_images']) == 0 and
            len(self.stats['missing_labels']) == 0 and
            len(self.stats['invalid_labels']) == 0 and
            len(self.stats['out_of_bounds_boxes']) == 0 and
            len(self.stats['zero_area_boxes']) == 0 and
            len(self.stats['cross_split_leakage']) == 0 and
            self.stats['total_instances'] > 0
        )
        
        report.append("-" * 70)
        report.append(f"AUDIT STATUS: {'PASSED (Dataset is Ready)' if is_pass else 'ISSUES DETECTED'}")
        report.append("=" * 70)
        return "\n".join(report)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "yolov8/data"
    validator = DatasetValidator(data_dir)
    validator.run_audit()
    print(validator.generate_report())
