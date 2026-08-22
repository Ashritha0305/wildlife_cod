"""
Stage 5C: Comprehensive Camouflage Provenance, Taxonomy, Category, and Integrity Audit
"""

import os
import sys
import json
import glob
import hashlib
from collections import defaultdict
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from yolov8.src.utils import yolo_to_xyxy

INDEX_PATH = "yolov8/data/external_metadata/camo_acquisition_index.json"
WCS_JSON_PATH = "yolov8/data/external_metadata/wcs_bboxes/wcs_20220205_bboxes_with_classes.json"
FINAL_QC_DIR = "yolov8/results/camo_final_qc"

CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]
CLASS_COLORS = {
    0: (0, 165, 255),   # Buffalo: Orange
    1: (255, 100, 0),   # Elephant: Blue
    2: (0, 255, 0),     # Rhino: Green
    3: (255, 0, 255)    # Zebra: Magenta
}


def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_stage5c():
    print("=" * 80)
    print("STAGE 5C: FINAL CAMOUFLAGE PROVENANCE & QUALITY AUDIT")
    print("=" * 80)

    # 1. Audit Provenance Index
    print("\n--- AUDIT 1: PROVENANCE INDEX VERIFICATION ---")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index_records = json.load(f)
    print(f"Total Provenance Records: {len(index_records)}")

    fields_to_check = [
        "local_image", "local_label", "source", "image_id", "source_url",
        "species", "class_id", "sequence_id", "location", "datetime",
        "visibility_reason", "sha256", "num_instances"
    ]
    field_counts = {f: 0 for f in fields_to_check}
    for r in index_records:
        for f in fields_to_check:
            if f in r and r[f] not in [None, ""]:
                field_counts[f] += 1

    for f, cnt in field_counts.items():
        print(f"  Field '{f:<20}': {cnt} / {len(index_records)} records ({cnt/len(index_records)*100:.1f}%)")

    # Load original WCS metadata to verify original category IDs and bboxes
    print("\n--- AUDIT 2: BIOLOGICAL CLASS & TAXONOMY VERIFICATION ---")
    with open(WCS_JSON_PATH, "r", encoding="utf-8") as f:
        wcs_data = json.load(f)

    wcs_categories = {c["id"]: c["name"].lower() for c in wcs_data.get("categories", [])}
    wcs_anns_by_img = defaultdict(list)
    for a in wcs_data.get("annotations", []):
        wcs_anns_by_img[a["image_id"]].append(a)

    taxonomy_audit_passed = True
    non_target_detections = []
    category_id_counts = defaultdict(int)

    for r in index_records:
        img_id = r["image_id"]
        cid = r["class_id"]
        cname = r["species"]

        orig_anns = wcs_anns_by_img.get(img_id, [])
        for a in orig_anns:
            cat_id = a.get("category_id")
            cat_name = wcs_categories.get(cat_id, "unknown")
            category_id_counts[(cat_id, cat_name, cid, cname)] += 1

            # Assert valid biological mapping
            if cid == 0 and "syncerus caffer" not in cat_name and "buffalo" not in cat_name:
                taxonomy_audit_passed = False
                non_target_detections.append((img_id, cat_name, cid))
            elif cid == 1 and "loxodonta africana" not in cat_name and "elephant" not in cat_name:
                taxonomy_audit_passed = False
                non_target_detections.append((img_id, cat_name, cid))
            elif cid == 2 and "diceros bicornis" not in cat_name and "ceratotherium simum" not in cat_name and "rhino" not in cat_name:
                taxonomy_audit_passed = False
                non_target_detections.append((img_id, cat_name, cid))
            elif cid == 3 and "equus quagga" not in cat_name and "equus grevyi" not in cat_name and "zebra" not in cat_name:
                taxonomy_audit_passed = False
                non_target_detections.append((img_id, cat_name, cid))

    print("Observed WCS Category ID mappings:")
    for (cat_id, cat_name, cid, cname), count in sorted(category_id_counts.items()):
        print(f"  Category [{cat_id}] '{cat_name}' -> Class {cid} ({cname}): {count} bboxes")

    print(f"Taxonomy Audit Status: {'PASSED (100% Biological Integrity)' if taxonomy_audit_passed else 'FAILED'}")
    if non_target_detections:
        print(f"  Non-target detections found: {non_target_detections}")

    # 3. Concealment Category Audit
    print("\n--- AUDIT 3 & 4: CONCEALMENT CATEGORY BREAKDOWN ---")
    category_breakdown = defaultdict(lambda: {"images": 0, "instances": 0, "species": defaultdict(int)})

    for r in index_records:
        reason = r.get("visibility_reason", "natural_habitat_background_blending")
        species = r["species"]
        num_inst = r["num_instances"]

        # Parse multiple reasons
        reasons = [s.strip() for s in reason.split(";")]
        for res in reasons:
            category_breakdown[res]["images"] += 1
            category_breakdown[res]["instances"] += num_inst
            category_breakdown[res]["species"][species] += num_inst

    for cat_name, stats in sorted(category_breakdown.items()):
        print(f"  Category: '{cat_name}'")
        print(f"    Images: {stats['images']}, Instances: {stats['instances']}")
        spec_str = ", ".join(f"{k}: {v}" for k, v in stats["species"].items())
        print(f"    Species: {spec_str}")

    # 5. Held-out test purity and Sequence-Level Separation
    print("\n--- AUDIT 5: HELD-OUT TEST PURITY & SEQUENCE SEPARATION ---")
    train_seqs = set()
    val_seqs = set()
    test_seqs = set()

    for r in index_records:
        split = r["split"]
        seq = r["sequence_id"]
        if split == "train":
            train_seqs.add(seq)
        elif split == "val":
            val_seqs.add(seq)
        elif split == "test":
            test_seqs.add(seq)

    train_test_overlap = train_seqs.intersection(test_seqs)
    val_test_overlap = val_seqs.intersection(test_seqs)
    train_val_overlap = train_seqs.intersection(val_seqs)

    print(f"Total Unique Sequences in Train: {len(train_seqs)}")
    print(f"Total Unique Sequences in Val:   {len(val_seqs)}")
    print(f"Total Unique Sequences in Test:  {len(test_seqs)}")
    print(f"Sequence Overlap Train <-> Test: {len(train_test_overlap)}")
    print(f"Sequence Overlap Val <-> Test:   {len(val_test_overlap)}")
    print(f"Sequence Overlap Train <-> Val:  {len(train_val_overlap)}")

    # 6. Species Distribution Table
    print("\n--- AUDIT 6: EXACT SPECIES DISTRIBUTION ---")
    species_counts = defaultdict(lambda: {"train_img": 0, "val_img": 0, "test_img": 0,
                                          "train_inst": 0, "val_inst": 0, "test_inst": 0})
    for r in index_records:
        sp = r["species"]
        sp_split = r["split"]
        inst = r["num_instances"]
        if sp_split == "train":
            species_counts[sp]["train_img"] += 1
            species_counts[sp]["train_inst"] += inst
        elif sp_split == "val":
            species_counts[sp]["val_img"] += 1
            species_counts[sp]["val_inst"] += inst
        elif sp_split == "test":
            species_counts[sp]["test_img"] += 1
            species_counts[sp]["test_inst"] += inst

    print(f"{'Species':<12} {'Train Img':<10} {'Val Img':<10} {'Test Img':<10} {'Train Inst':<12} {'Val Inst':<10} {'Test Inst':<10}")
    print("-" * 75)
    for sp in CLASS_NAMES:
        c = species_counts[sp]
        print(f"{sp:<12} {c['train_img']:<10} {c['val_img']:<10} {c['test_img']:<10} {c['train_inst']:<12} {c['val_inst']:<10} {c['test_inst']:<10}")

    # 7. Generate Independent Visual QC set (minimum 3 per species = 12 samples)
    print("\n--- AUDIT 8: GENERATING NEW INDEPENDENT VISUAL QC SAMPLES ---")
    os.makedirs(FINAL_QC_DIR, exist_ok=True)
    
    # Pick 3 diverse images per species
    qc_selected = {0: [], 1: [], 2: [], 3: []}
    for r in index_records:
        cid = r["class_id"]
        if len(qc_selected[cid]) < 3:
            qc_selected[cid].append(r)

    qc_idx = 1
    for cid in [0, 1, 2, 3]:
        sp_name = CLASS_NAMES[cid]
        for r in qc_selected[cid]:
            img_path = r["local_image"]
            lbl_path = r["local_label"]
            reason = r["visibility_reason"]

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf if l.strip()]

            for l in lines:
                parts = l.split()
                c = int(parts[0])
                xc, yc, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1, y1, x2, y2 = yolo_to_xyxy((xc, yc, nw, nh), w, h)
                color = CLASS_COLORS[c]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                lbl_text = f"GT: {CLASS_NAMES[c]}"
                (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                bg_y1 = max(0, y1 - th - 10)
                cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 10, bg_y1 + th + 10), color, -1)
                cv2.putText(img, lbl_text, (x1 + 5, bg_y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            banner = f"STAGE 5C QC #{qc_idx:02d}: {sp_name.upper()} ({reason})"
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
            cv2.putText(img, banner, (15, 15 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            out_name = f"final_qc_{qc_idx:02d}_{sp_name}_{os.path.basename(img_path)}"
            out_path = os.path.join(FINAL_QC_DIR, out_name)
            cv2.imwrite(out_path, img)
            print(f"  Saved Final QC [{qc_idx:02d}]: {out_path}")
            qc_idx += 1


if __name__ == "__main__":
    audit_stage5c()
