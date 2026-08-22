"""
Stage 6A: YOLOv8s Camouflage-Aware Model B Training Script
Trains YOLOv8s initialized from official pretrained weights (yolov8s.pt) on the combined
baseline + genuine camera-trap camouflage training dataset.
"""

import os
import sys
import torch
from ultralytics import YOLO

def train_model_b():
    print("=" * 80)
    print("STAGE 6A: STARTING YOLOv8s CAMOUFLAGE-AWARE MODEL B TRAINING")
    print("=" * 80)

    # 1. Device Verification
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM:    {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # 2. Config & Paths
    data_cfg = os.path.abspath("yolov8/config/camouflage.yaml")
    project_dir = os.path.abspath("yolov8/results")
    exp_name = "camouflage_yolov8s"
    pretrained_model = "yolov8s.pt"

    print(f"\nTraining Configuration:")
    print(f"  Pretrained Weights: {pretrained_model}")
    print(f"  Dataset Config:     {data_cfg}")
    print(f"  Output Directory:   {os.path.join(project_dir, exp_name)}")
    print(f"  Epochs:             20")
    print(f"  Batch Size:         16")
    print(f"  Image Size:         640")
    print(f"  Random Seed:        42")
    print(f"  Device:             {device}")
    print("=" * 80)

    # 3. Initialize Model B from official pretrained weights
    model = YOLO(pretrained_model)

    # 4. Run Training
    results = model.train(
        data=data_cfg,
        epochs=20,
        batch=16,
        imgsz=640,
        device=0 if device.startswith("cuda") else "cpu",
        seed=42,
        project=project_dir,
        name=exp_name,
        exist_ok=True,
        workers=4,
        val=True,
        plots=True,
        save=True
    )

    print("\n" + "=" * 80)
    print("MODEL B TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Best Weights Path: {os.path.join(project_dir, exp_name, 'weights', 'best.pt')}")
    print("=" * 80)

if __name__ == "__main__":
    train_model_b()
