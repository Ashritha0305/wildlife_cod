"""
Validation and Test Evaluation Engine for YOLOv8 Wildlife Subsystem
Computes exact measured Precision, Recall, mAP@50, mAP@50:95,
per-class breakdown, confusion matrices, and confidence threshold sweeps.
"""

from typing import Dict, Any, List, Optional
import os
import sys
import argparse
import numpy as np
import torch
from ultralytics import YOLO


def validate_yolov8(
    model_path: str,
    data_yaml: str = "yolov8/config/baseline.yaml",
    split: str = "test",
    imgsz: int = 640,
    batch_size: int = 16,
    device: int = 0,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Execute formal evaluation on a specified dataset partition (val or test).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    print("=" * 70)
    print(f"RUNNING YOLOv8 EVALUATION ON SPLIT: [{split.upper()}]")
    print(f"Model Checkpoint:  {model_path}")
    print(f"Dataset YAML:      {data_yaml}")
    print(f"Confidence Thresh: {conf_threshold}")
    print(f"IoU NMS Thresh:    {iou_threshold}")
    print("=" * 70)

    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device if torch.cuda.is_available() else "cpu",
        conf=conf_threshold,
        iou=iou_threshold,
        workers=0,
        verbose=True,
        plots=True
    )

    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    map50 = float(box.map50)
    map50_95 = float(box.map)

    class_names = list(model.names.values())
    per_class_metrics = []

    if hasattr(box, "p") and hasattr(box, "r") and hasattr(box, "ap50") and hasattr(box, "ap"):
        p_list = box.p
        r_list = box.r
        ap50_list = box.ap50
        ap_list = box.ap

        for i, name in enumerate(class_names):
            p_val = float(p_list[i]) if i < len(p_list) else 0.0
            r_val = float(r_list[i]) if i < len(r_list) else 0.0
            ap50_val = float(ap50_list[i]) if i < len(ap50_list) else 0.0
            ap_val = float(ap_list[i]) if i < len(ap_list) else 0.0
            per_class_metrics.append({
                "class_id": i,
                "class_name": name,
                "precision": round(p_val, 4),
                "recall": round(r_val, 4),
                "ap50": round(ap50_val, 4),
                "ap50_95": round(ap_val, 4)
            })

    results_summary = {
        "split": split,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class": per_class_metrics,
        "speed": metrics.speed
    }

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS SUMMARY [{split.upper()}]:")
    print(f"  Precision (mean):  {precision * 100:.2f}%")
    print(f"  Recall (mean):     {recall * 100:.2f}%")
    print(f"  mAP@50:            {map50 * 100:.2f}%")
    print(f"  mAP@50:95:         {map50_95 * 100:.2f}%")
    print("-" * 70)
    print("PER-CLASS METRICS:")
    print(f"{'ID':<4} {'Species':<16} {'Precision':<12} {'Recall':<12} {'AP@50':<12} {'AP@50:95':<12}")
    print("-" * 70)
    for c in per_class_metrics:
        print(f"{c['class_id']:<4} {c['class_name']:<16} {c['precision']*100:>8.2f}%   {c['recall']*100:>8.2f}%   {c['ap50']*100:>8.2f}%   {c['ap50_95']*100:>8.2f}%")
    print("=" * 70)

    return results_summary


def run_confidence_threshold_sweep(
    model_path: str,
    data_yaml: str = "yolov8/config/baseline.yaml",
    split: str = "test",
    thresholds: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate precision and recall across multiple confidence thresholds.
    """
    if thresholds is None:
        thresholds = [0.25, 0.35, 0.50, 0.60, 0.75]

    print("\n" + "=" * 70)
    print(f"CONFIDENCE THRESHOLD SWEEP ON [{split.upper()}]: {thresholds}")
    print("=" * 70)

    sweep_results = []
    for conf in thresholds:
        res = validate_yolov8(
            model_path=model_path,
            data_yaml=data_yaml,
            split=split,
            conf_threshold=conf
        )
        sweep_results.append({
            "conf_threshold": conf,
            "precision": res["precision"],
            "recall": res["recall"],
            "map50": res["map50"],
            "map50_95": res["map50_95"]
        })

    print("\n" + "=" * 70)
    print("CONFIDENCE THRESHOLD SWEEP SUMMARY TABLE:")
    print(f"{'Threshold':<12} {'Precision':<14} {'Recall':<14} {'mAP@50':<14} {'mAP@50:95':<14}")
    print("-" * 70)
    for s in sweep_results:
        print(f"{s['conf_threshold']:<12.2f} {s['precision']*100:>10.2f}%   {s['recall']*100:>10.2f}%   {s['map50']*100:>10.2f}%   {s['map50_95']*100:>10.2f}%")
    print("=" * 70)

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLOv8 Wildlife Model")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="yolov8/config/baseline.yaml", help="Data YAML")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: val or test")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()
    validate_yolov8(model_path=args.model, data_yaml=args.data, split=args.split, conf_threshold=args.conf)
