"""
Stage 6A: Generate Representative Error Analysis Visualizations
Creates side-by-side comparison images (Ground Truth vs Model A vs Model B) across 6 representative camouflage scenarios:
  1. Dense Vegetation Concealment
  2. Night / Twilight IR Low-Light
  3. Herd Occlusion / Partial Overlap
  4. Mud / Dust Background Blending
  5. Small / Distant Concealed Target
  6. True Negative Cluttered Background
"""

import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_A_PATH = "yolov8/results/baseline_yolov8s/weights/best.pt"
MODEL_B_PATH = "yolov8/results/camouflage_yolov8s/weights/best.pt"
CAMO_IMG_DIR = "yolov8/data/camo_benchmark/4class_camo/images"
CAMO_LBL_DIR = "yolov8/data/camo_benchmark/4class_camo/labels"
NEG_IMG_DIR = "yolov8/data/camo_benchmark/negatives/images"
OUTPUT_DIR = "yolov8/results/camouflage_yolov8s_error_analysis"

CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]
CLASS_COLORS = {
    "buffalo": (0, 165, 255),    # Orange
    "elephant": (0, 255, 255),   # Yellow
    "rhino": (255, 0, 255),      # Magenta
    "zebra": (0, 255, 0)         # Green
}


def draw_boxes_on_image(img, boxes, labels, confidences=None, title=""):
    canvas = img.copy()
    h, w = canvas.shape[:2]
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cname = labels[idx]
        color = CLASS_COLORS.get(cname, (255, 255, 255))
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        conf_str = f" {confidences[idx]:.2f}" if confidences is not None else ""
        text = f"{cname}{conf_str}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(canvas, text, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    # Add title banner
    cv2.rectangle(canvas, (0, 0), (w, 32), (30, 30, 30), -1)
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def read_yolo_ground_truth(lbl_path, img_shape):
    if not os.path.exists(lbl_path):
        return [], []
    h, w = img_shape[:2]
    boxes, labels = [], []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_NAMES[cls_id])
    return boxes, labels


def get_model_detections(model, img, conf=0.25):
    res = model.predict(img, imgsz=640, conf=conf, device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
    boxes, labels, confs = [], [], []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        conf_vals = res.boxes.conf.cpu().numpy()
        for b, c, cf in zip(xyxy, cls_ids, conf_vals):
            boxes.append(b.tolist())
            labels.append(CLASS_NAMES[c])
            confs.append(float(cf))
    return boxes, labels, confs


def generate_error_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_a = YOLO(MODEL_A_PATH)
    model_b = YOLO(MODEL_B_PATH)

    # Candidate images
    camo_images = sorted(glob.glob(os.path.join(CAMO_IMG_DIR, "*.*")))
    neg_images = sorted(glob.glob(os.path.join(NEG_IMG_DIR, "*.*")))

    # Select representative samples
    selected_indices = [0, 4, 10, 16, 25, 34, 40]
    eval_samples = [camo_images[i] for i in selected_indices if i < len(camo_images)]
    if len(neg_images) > 0:
        eval_samples.append(neg_images[0])
        eval_samples.append(neg_images[5])

    print(f"Generating side-by-side error analysis for {len(eval_samples)} representative cases...")

    for idx, img_path in enumerate(eval_samples):
        bname = os.path.basename(img_path)
        stem = os.path.splitext(bname)[0]
        img = cv2.imread(img_path)
        if img is None:
            continue

        lbl_path = os.path.join(CAMO_LBL_DIR, stem + ".txt")
        gt_boxes, gt_labels = read_yolo_ground_truth(lbl_path, img.shape)

        a_boxes, a_labels, a_confs = get_model_detections(model_a, img)
        b_boxes, b_labels, b_confs = get_model_detections(model_b, img)

        # Draw 3 panels
        p_gt = draw_boxes_on_image(img, gt_boxes, gt_labels, title=f"Ground Truth ({len(gt_boxes)} targets)")
        p_a = draw_boxes_on_image(img, a_boxes, a_labels, a_confs, title=f"Model A: Baseline YOLOv8s ({len(a_boxes)} det)")
        p_b = draw_boxes_on_image(img, b_boxes, b_labels, b_confs, title=f"Model B: Camo-Aware YOLOv8s ({len(b_boxes)} det)")

        # Target height
        target_h = 480
        scale = target_h / img.shape[0]
        target_w = int(img.shape[1] * scale)

        p_gt_r = cv2.resize(p_gt, (target_w, target_h))
        p_a_r = cv2.resize(p_a, (target_w, target_h))
        p_b_r = cv2.resize(p_b, (target_w, target_h))

        triptych = np.hstack([p_gt_r, p_a_r, p_b_r])

        out_path = os.path.join(OUTPUT_DIR, f"error_analysis_case_{idx+1:02d}_{stem}.jpg")
        cv2.imwrite(out_path, triptych)
        print(f"  Saved: {out_path}")

    print(f"\nAll error analysis visual comparisons saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_error_analysis()
