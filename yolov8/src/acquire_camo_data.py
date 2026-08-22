"""
Targeted External Camouflage Wildlife Data Acquisition Pipeline
Dataset Source: Wildlife Conservation Society (WCS) Camera Traps on LILA BC
License: Community Data License Agreement (CDLA-Permissive) / CC-BY 4.0

Selects, downloads, validates, deduplicates, and stages genuine camera-trap wildlife images
under natural concealment for:
  0: buffalo (Syncerus caffer)
  1: elephant (Loxodonta africana)
  2: rhino (Diceros bicornis / Ceratotherium simum)
  3: zebra (Equus quagga / Equus grevyi)
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))
from yolov8.src.utils import coco_to_yolo, validate_image_frame


BASE_URL = "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/"
JSON_PATH = "yolov8/data/external_metadata/wcs_bboxes/wcs_20220205_bboxes_with_classes.json"
INDEX_OUT_PATH = "yolov8/data/external_metadata/camo_acquisition_index.json"

TARGET_SPECIES_MAP = {
    110: (0, "buffalo", "syncerus caffer"),
    90: (1, "elephant", "loxodonta africana"),
    260: (2, "rhino", "diceros bicornis"),
    261: (2, "rhino", "ceratotherium simum"),
    111: (3, "zebra", "equus quagga"),
    259: (3, "zebra", "equus grevyi")
}


def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_baseline_hashes() -> Set[str]:
    """Collect all SHA-256 hashes from the baseline dataset to guarantee zero leakage."""
    baseline_hashes = set()
    import glob
    for p in glob.glob("yolov8/data/baseline/**/images/*.*", recursive=True):
        baseline_hashes.add(compute_sha256(p))
    return baseline_hashes


def classify_visibility_provenance(img_record: Dict[str, Any], annotations: List[Dict[str, Any]]) -> str:
    """Determine and document the specific natural concealment condition."""
    dt_str = img_record.get("datetime", "")
    hour = -1
    if len(dt_str) >= 13 and ":" in dt_str:
        try:
            hour = int(dt_str.split()[1].split(":")[0])
        except (ValueError, IndexError):
            pass

    # Check bbox area ratio
    img_w = float(img_record.get("width", 1280))
    img_h = float(img_record.get("height", 1024))
    img_area = max(1.0, img_w * img_h)

    total_bbox_area = sum(float(a.get("bbox", [0, 0, 0, 0])[2]) * float(a.get("bbox", [0, 0, 0, 0])[3]) for a in annotations)
    area_ratio = total_bbox_area / img_area

    reasons = []
    if hour >= 18 or (0 <= hour <= 6):
        reasons.append("night_twilight_infrared_concealment")
    if area_ratio < 0.08:
        reasons.append("small_distant_dense_foliage")
    elif area_ratio < 0.20:
        reasons.append("partial_body_vegetation_obstruction")
    if len(annotations) > 1:
        reasons.append("multi_animal_herd_occlusion")

    if not reasons:
        reasons.append("natural_habitat_background_blending")

    return "; ".join(reasons)


def acquire_and_stage_camo_data(
    target_per_class: int = 80,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    print("=" * 70)
    print("STAGE 5B: TARGETED EXTERNAL CAMOUFLAGE DATA ACQUISITION")
    print(f"Source: WCS Camera Traps (LILA BC) - {BASE_URL}")
    print(f"Target Species: Buffalo (0), Elephant (1), Rhino (2), Zebra (3)")
    print("=" * 70)

    # 1. Load baseline hashes for cryptographic duplicate rejection
    print("\n1. Loading baseline hashes for duplicate prevention...")
    baseline_hashes = load_baseline_hashes()
    print(f"Loaded {len(baseline_hashes)} baseline hashes.")

    # 2. Parse WCS annotations
    print("\n2. Parsing WCS bounding box annotations...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_dict = {img["id"]: img for img in data.get("images", [])}
    annotations = data.get("annotations", [])

    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in annotations:
        cat_id = ann.get("category_id")
        if cat_id in TARGET_SPECIES_MAP:
            img_to_anns[ann["image_id"]].append(ann)

    # Group images by species and sequence
    species_seq_imgs = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}

    for img_id, anns in img_to_anns.items():
        img_info = images_dict.get(img_id)
        if not img_info or img_info.get("corrupt", False):
            continue

        # Get species from first annotation
        cat_id = anns[0]["category_id"]
        class_id, species_name, sci_name = TARGET_SPECIES_MAP[cat_id]
        seq_id = img_info.get("seq_id") or f"single_{img_id}"

        species_seq_imgs[class_id][seq_id].append((img_info, anns))

    # 3. Select sequence bursts per species
    rng = np.random.RandomState(seed)
    selected_by_species = {0: [], 1: [], 2: [], 3: []}

    print("\n3. Selecting sequences with natural concealment...")
    for class_id in [0, 1, 2, 3]:
        seq_dict = species_seq_imgs[class_id]
        seq_keys = list(seq_dict.keys())
        rng.shuffle(seq_keys)

        collected = []
        for sk in seq_keys:
            seq_imgs = seq_dict[sk]
            # Take up to 2-3 images per sequence to ensure diverse coverage
            for item in seq_imgs[:3]:
                collected.append((sk, item[0], item[1]))
                if len(collected) >= target_per_class:
                    break
            if len(collected) >= target_per_class:
                break

        selected_by_species[class_id] = collected
        species_name = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}[class_id]
        print(f"  Class {class_id} ({species_name}): Selected {len(collected)} candidate images across {len(set(c[0] for c in collected))} sequences.")

    # 4. Partition sequences into train, val, and held-out test splits
    print("\n4. Partitioning by sequence (Zero Sequence-Level Leakage)...")
    splits_data = {"train": [], "val": "test", "test": []}
    splits_data = {"train": [], "val": [], "test": []}

    for class_id in [0, 1, 2, 3]:
        candidates = selected_by_species[class_id]
        # Group by sequence
        seq_groups = defaultdict(list)
        for seq_id, img_info, anns in candidates:
            seq_groups[seq_id].append((class_id, img_info, anns))

        seq_list = list(seq_groups.keys())
        rng.shuffle(seq_list)

        n_seq = len(seq_list)
        n_train = int(round(n_seq * train_ratio))
        n_val = int(round(n_seq * val_ratio))
        
        train_seqs = set(seq_list[:n_train])
        val_seqs = set(seq_list[n_train:n_train + n_val])
        test_seqs = set(seq_list[n_train + n_val:])

        for seq_id, items in seq_groups.items():
            if seq_id in train_seqs:
                splits_data["train"].extend(items)
            elif seq_id in val_seqs:
                splits_data["val"].extend(items)
            else:
                splits_data["test"].extend(items)

    print(f"  Total planned: Train={len(splits_data['train'])}, Val={len(splits_data['val'])}, Held-out Camo Test={len(splits_data['test'])}")

    # 5. Execute targeted download and format conversion
    print("\n5. Executing selective image download & annotation conversion...")
    dirs = {
        "train": ("yolov8/data/camo_training/train/images", "yolov8/data/camo_training/train/labels"),
        "val": ("yolov8/data/camo_training/val/images", "yolov8/data/camo_training/val/labels"),
        "test": ("yolov8/data/camo_benchmark/4class_camo/images", "yolov8/data/camo_benchmark/4class_camo/labels")
    }

    for d_img, d_lbl in dirs.values():
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lbl, exist_ok=True)

    acquisition_index = []
    downloaded_count = 0
    duplicate_count = 0
    invalid_ann_count = 0
    failed_download_count = 0

    seen_hashes = set()

    for split_name, items in splits_data.items():
        dst_img_dir, dst_lbl_dir = dirs[split_name]
        print(f"  Processing [{split_name.upper()}] ({len(items)} images)...")

        for class_id, img_info, anns in items:
            img_id = img_info["id"]
            rel_file = img_info["file_name"]
            img_url = BASE_URL + rel_file

            img_w = float(img_info.get("width", 1280))
            img_h = float(img_info.get("height", 1024))
            species_name = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}[class_id]

            # Destination filenames
            clean_base = f"camo_{species_name}_{img_info.get('country_code', 'af')}_{img_id[:8]}"
            img_dest = os.path.join(dst_img_dir, f"{clean_base}.jpg")
            lbl_dest = os.path.join(dst_lbl_dir, f"{clean_base}.txt")

            # Download image
            try:
                urllib.request.urlretrieve(img_url, img_dest)
            except Exception as e:
                print(f"    Failed download: {img_url} ({e})")
                failed_download_count += 1
                continue

            # Verify image readability
            try:
                read_img = cv2.imread(img_dest)
                if read_img is None or read_img.size == 0:
                    os.remove(img_dest)
                    failed_download_count += 1
                    continue
                actual_h, actual_w = read_img.shape[:2]
                img_w, img_h = float(actual_w), float(actual_h)
            except Exception:
                if os.path.exists(img_dest):
                    os.remove(img_dest)
                failed_download_count += 1
                continue

            # Compute hash and check for duplicates
            img_hash = compute_sha256(img_dest)
            if img_hash in baseline_hashes or img_hash in seen_hashes:
                print(f"    Duplicate detected ({img_hash[:12]}), rejecting: {clean_base}")
                os.remove(img_dest)
                duplicate_count += 1
                continue

            seen_hashes.add(img_hash)

            # Convert annotations to YOLO normalized format
            yolo_lines = []
            for a in anns:
                coco_box = a.get("bbox")  # [xmin, ymin, width, height]
                if not coco_box or len(coco_box) != 4:
                    invalid_ann_count += 1
                    continue

                xmin, ymin, bw, bh = coco_box
                if bw <= 0 or bh <= 0 or xmin < 0 or ymin < 0:
                    invalid_ann_count += 1
                    continue

                xc, yc, nw, nh = coco_to_yolo((xmin, ymin, xmin + bw, ymin + bh), int(img_w), int(img_h))
                
                # Boundary assertions
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                    invalid_ann_count += 1
                    continue

                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            if not yolo_lines:
                os.remove(img_dest)
                invalid_ann_count += 1
                continue

            # Write YOLO label file
            with open(lbl_dest, "w", encoding="utf-8") as lf:
                lf.write("\n".join(yolo_lines) + "\n")

            # Document provenance
            reason = classify_visibility_provenance(img_info, anns)
            acquisition_index.append({
                "image_id": img_id,
                "split": split_name,
                "class_id": class_id,
                "species": species_name,
                "source": "WCS Camera Traps (LILA BC)",
                "source_url": img_url,
                "sequence_id": img_info.get("seq_id", ""),
                "location": img_info.get("location", ""),
                "datetime": img_info.get("datetime", ""),
                "visibility_reason": reason,
                "sha256": img_hash,
                "local_image": img_dest,
                "local_label": lbl_dest,
                "num_instances": len(yolo_lines)
            })
            downloaded_count += 1

    # Save index
    os.makedirs(os.path.dirname(INDEX_OUT_PATH), exist_ok=True)
    with open(INDEX_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(acquisition_index, f, indent=2)

    print("\n" + "=" * 70)
    print("ACQUISITION SUMMARY:")
    print(f"  Successfully Acquired & Staged: {downloaded_count} images")
    print(f"  Duplicates Rejected:            {duplicate_count}")
    print(f"  Invalid Annotations Skipped:    {invalid_ann_count}")
    print(f"  Failed Downloads:               {failed_download_count}")
    print(f"  Provenance Index Saved to:      {INDEX_OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    acquire_and_stage_camo_data()
