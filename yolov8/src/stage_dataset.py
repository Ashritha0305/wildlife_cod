"""
Dataset Staging & Global Deduplication Script for YOLOv8 Wildlife Subsystem
Ensures zero data leakage and 100% disjoint splits across train, val, and test partitions.
"""

import os
import sys
import shutil
import zipfile
import hashlib
import glob
import urllib.request
from typing import Dict, List, Tuple, Set


def download_file(url: str, dest_path: str) -> str:
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return dest_path
        
    print(f"Downloading from {url} to {dest_path}...")
    
    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rDownloading: {percent}% ({count * block_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()
        
    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print("\nDownload complete.")
    return dest_path


def extract_zip(zip_path: str, extract_dir: str) -> str:
    """Extract a ZIP archive into a destination directory."""
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    return extract_dir


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def enforce_strict_split_disjointness(data_root: str):
    """
    Enforce global hash uniqueness:
    1. Seen in train -> remove from val and test.
    2. Seen in val -> remove from test.
    3. Duplicate within same split -> remove extra copies.
    """
    print("\n--- Enforcing Global Split Disjointness (Zero Leakage) ---")
    seen_hashes: Dict[str, Tuple[str, str]] = {}
    removed_count = 0
    
    # Priority order: train -> val -> test
    for split in ["train", "val", "test"]:
        s_img_dir = os.path.join(data_root, split, "images")
        s_lbl_dir = os.path.join(data_root, split, "labels")
        
        img_files = sorted(list(glob.glob(os.path.join(s_img_dir, "*.*"))))
        for img_p in img_files:
            h = compute_file_hash(img_p)
            if h in seen_hashes:
                origin_split, origin_file = seen_hashes[h]
                base = os.path.splitext(os.path.basename(img_p))[0]
                lbl_p = os.path.join(s_lbl_dir, f"{base}.txt")
                
                os.remove(img_p)
                if os.path.exists(lbl_p):
                    os.remove(lbl_p)
                print(f"Removed duplicate in [{split}]: {os.path.basename(img_p)} (Original in [{origin_split}])")
                removed_count += 1
            else:
                seen_hashes[h] = (split, img_p)
                
    print(f"Disjointness complete. Total duplicates purged: {removed_count}. Unique dataset images: {len(seen_hashes)}.")


def stage_african_wildlife(raw_extract_dir: str, target_data_root: str):
    splits = ["train", "val", "test"]
    source_root = raw_extract_dir
    if os.path.exists(os.path.join(raw_extract_dir, "african-wildlife")):
        source_root = os.path.join(raw_extract_dir, "african-wildlife")
        
    print(f"Staging African Wildlife from: {source_root}")
    
    for split in splits:
        src_img_dir = os.path.join(source_root, split, "images")
        src_lbl_dir = os.path.join(source_root, split, "labels")
        
        dst_img_dir = os.path.join(target_data_root, split, "images")
        dst_lbl_dir = os.path.join(target_data_root, split, "labels")
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if os.path.exists(src_img_dir):
            for f in os.listdir(src_img_dir):
                shutil.copy2(os.path.join(src_img_dir, f), os.path.join(dst_img_dir, f))
                    
        if os.path.exists(src_lbl_dir):
            for f in os.listdir(src_lbl_dir):
                shutil.copy2(os.path.join(src_lbl_dir, f), os.path.join(dst_lbl_dir, f))
                
    enforce_strict_split_disjointness(target_data_root)


if __name__ == "__main__":
    raw_dir = "yolov8/data/raw"
    target_dir = "yolov8/data"
    
    zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
    zip_path = os.path.join(raw_dir, "african-wildlife.zip")
    
    try:
        download_file(zip_url, zip_path)
        extract_dir = os.path.join(raw_dir, "african-wildlife-extracted")
        if not os.path.exists(extract_dir):
            extract_zip(zip_path, extract_dir)
        stage_african_wildlife(extract_dir, target_dir)
    except Exception as e:
        print(f"Error staging dataset: {e}")
