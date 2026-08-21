"""
Training Pipeline for YOLOv8 Wildlife Object Detection Subsystem
Supports transfer learning, configurable hyperparameters, GPU device mapping,
and experiment artifact management.
"""

from typing import Dict, Any, Optional
import os
import sys
import time
import argparse
import yaml
import torch
from ultralytics import YOLO


def train_yolov8(
    model_type: str = "yolov8n.pt",
    data_yaml: str = "yolov8/config/baseline.yaml",
    epochs: int = 25,
    batch_size: int = 16,
    imgsz: int = 640,
    device: int = 0,
    project: str = "yolov8/results",
    name: str = "baseline_yolov8n",
    seed: int = 42,
    patience: int = 10,
    lr0: float = 0.01,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    workers: int = 4
) -> Dict[str, Any]:
    """
    Train a YOLOv8 model using transfer learning from official pretrained weights.
    """
    data_yaml = os.path.abspath(data_yaml)
    project = os.path.abspath(project)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data configuration YAML not found: {data_yaml}")

    print("=" * 70)
    print(f"STARTING YOLOv8 TRAINING: {name}")
    print(f"Model Variant:      {model_type}")
    print(f"Dataset YAML:       {data_yaml}")
    print(f"Epochs:             {epochs}")
    print(f"Batch Size:         {batch_size}")
    print(f"Image Resolution:   {imgsz}x{imgsz}")
    print(f"Device:             CUDA Device {device}" if torch.cuda.is_available() else "CPU")
    print(f"Random Seed:        {seed}")
    print(f"Output Directory:   {os.path.join(project, name)}")
    print("=" * 70)

    # Initialize pretrained YOLOv8 model for transfer learning
    model = YOLO(model_type)

    start_time = time.perf_counter()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device if torch.cuda.is_available() else "cpu",
        project=project,
        name=name,
        seed=seed,
        patience=patience,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        workers=workers,
        exist_ok=True,
        plots=True,
        verbose=True,
        save=True,
        val=True
    )

    elapsed_sec = time.perf_counter() - start_time
    save_dir = getattr(results, "save_dir", os.path.join(project, name))

    print("=" * 70)
    print(f"TRAINING COMPLETE: {name}")
    print(f"Total Duration:     {elapsed_sec / 60.0:.2f} minutes ({elapsed_sec:.1f}s)")
    print(f"Results Directory:  {save_dir}")
    print("=" * 70)

    return {
        "model_type": model_type,
        "name": name,
        "save_dir": str(save_dir),
        "duration_sec": round(elapsed_sec, 2),
        "best_checkpoint": os.path.join(save_dir, "weights", "best.pt"),
        "last_checkpoint": os.path.join(save_dir, "weights", "last.pt")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Wildlife Detector")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (yolov8n.pt or yolov8s.pt)")
    parser.add_argument("--data", type=str, default="yolov8/config/baseline.yaml", help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--name", type=str, default="baseline_yolov8n", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")

    args = parser.parse_args()

    train_yolov8(
        model_type=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        seed=args.seed,
        device=args.device
    )
