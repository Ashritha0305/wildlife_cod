"""
Ground-Truth Annotation Visualizer for YOLOv8 Wildlife Subsystem
Draws normalized YOLO bounding boxes on real dataset images to visually inspect
annotation alignment, class labels, camouflage instances, and occlusions before training.
"""

from typing import List, Optional
import os
import sys
import glob
import random
import cv2
from yolov8.src.utils import yolo_to_xyxy


CLASS_COLORS = [
    (0, 255, 0),     # Bright Green
    (0, 0, 255),     # Red
    (255, 165, 0),   # Orange
    (255, 255, 0),   # Cyan/Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (0, 128, 255),   # Light Orange
    (255, 192, 203), # Pink
    (128, 128, 0),   # Teal
    (0, 100, 0),     # Dark Green
    (75, 0, 130),    # Indigo
]


def draw_yolo_annotations(
    img_path: str,
    lbl_path: str,
    class_names: List[str],
    output_path: Optional[str] = None
) -> Optional[str]:
    """
    Read an image and its corresponding YOLO label file, draw ground truth boxes, and save/display.
    """
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        return None
        
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    h_img, w_img = img.shape[:2]
    
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
            
        cls_id = int(parts[0])
        x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        
        x1, y1, x2, y2 = yolo_to_xyxy((x_c, y_c, w, h), w_img, h_img)
        
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label header
        label_text = f"{cls_name} (ID:{cls_id})"
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - txt_h - 4)), (x1 + txt_w + 4, max(0, y1)), color, -1)
        cv2.putText(
            img,
            label_text,
            (x1 + 2, max(txt_h + 2, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, img)
        return output_path
        
    return None


def visualize_dataset_samples(
    data_root: str,
    class_names: List[str],
    output_dir: str = "yolov8/results/visualizations",
    samples_per_split: int = 4
) -> List[str]:
    """
    Randomly select sample images from train/val/test splits and render ground truth annotations.
    """
    saved_samples = []
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(data_root, split, "images")
        lbl_dir = os.path.join(data_root, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        img_files = glob.glob(os.path.join(img_dir, "*.*"))
        if not img_files:
            continue
            
        selected_imgs = random.sample(img_files, min(samples_per_split, len(img_files)))
        
        for idx, img_p in enumerate(selected_imgs):
            base = os.path.splitext(os.path.basename(img_p))[0]
            lbl_p = os.path.join(lbl_dir, f"{base}.txt")
            if os.path.exists(lbl_p):
                out_p = os.path.join(output_dir, f"{split}_sample_{idx+1}_{base}.jpg")
                saved = draw_yolo_annotations(img_p, lbl_p, class_names, out_p)
                if saved:
                    saved_samples.append(saved)
                    
    return saved_samples


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "yolov8/data"
    print(f"Visualizing samples from: {data_dir}")
