# Camouflage-Aware Wildlife Detection and Threat Analysis System

**Module:** YOLOv8 Wildlife Object Detection Subsystem  
**Lead Component:** Standalone YOLOv8s Wildlife Detection Engine  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM, CUDA 12.6)  
**Supported Target Classes:** `0: buffalo`, `1: elephant`, `2: rhino`, `3: zebra`  

---

## 1. System Architecture & High-Level Pipeline

This repository implements the **YOLOv8 wildlife object detection subsystem** within the complete multi-stage computer vision surveillance pipeline:

```text
Camera / Video Stream
       │
   ┌───┴───┐
   ↓       ↓
 U-Net   Optical Flow
 (Camo)  (Motion)
   └───┬───┘
       ↓
     Camouflage-Aware Fusion
       ↓
     YOLOv8s Wildlife Object Detection [THIS MODULE]
       ↓
     Structured Detections [Interface Boundary]
       ↓
   Deep SORT Multi-Object Tracking
       ↓
   ZoeDepth Metric Depth Estimation
       ↓
   Rule-Based + Fuzzy Logic Threat Analysis
       ↓
   Threat Level + Alert Visualization
```

---

## 2. Subsystem Scope & Architectural Boundaries

### In-Scope Responsibilities (YOLOv8 Subsystem):
- Real-time wildlife object detection on single images, image batches, and video streams.
- Bounding-box coordinate extraction clamped strictly within image boundaries ($0 \le x_1 < x_2 \le \text{width}$, $0 \le y_1 < y_2 \le \text{height}$).
- Biological species identification (`buffalo`, `elephant`, `rhino`, `zebra`) and confidence scoring.
- Structured detection output strictly conforming to the downstream Deep SORT contract.

### Explicit Out-of-Scope Responsibilities (Downstream Subsystems):
- **Multi-Object Tracking:** Handled exclusively by Deep SORT (assigns track IDs, Kalman state estimation, ReID).
- **Depth / Distance Estimation:** Handled exclusively by ZoeDepth (metric depth maps).
- **Threat Level Assessment:** Handled by Rule-Based & Fuzzy Logic subsystem.
- **Alert Dispatch:** Handled by Notification & Alert subsystem.
- **No Neural Fusion in YOLO:** YOLOv8s operates as a standalone object detector receiving standard $640 \times 640 \times 3$ BGR frames.

---

## 3. Authoritative Models & Results Summary

Under the primary operational evaluation protocol (`imgsz=640, conf=0.25, iou=0.60, batch=16, device=CUDA:0`):

| Model | Checkpoint Path | Normal Test mAP@50 | Camouflage Test mAP@50 | Negative FPPI | Throughput (RTX 4050) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Model A (Baseline)** | [`yolov8/results/baseline_yolov8s/weights/best.pt`](file:///c:/wildlife_COD/yolov8/results/baseline_yolov8s/weights/best.pt) | **95.80%** | 15.07% | 0.9000 | 90.89 FPS (11.00 ms) |
| **Model B (Camouflage)** | [`yolov8/results/camouflage_yolov8s/weights/best.pt`](file:///c:/wildlife_COD/yolov8/results/camouflage_yolov8s/weights/best.pt) | **95.23%** | **58.07%** | **0.8333** | **113.47 FPS (8.81 ms)** |
| **Measured Difference ($\Delta$)** | — | **-0.57 pp** | **+43.00 pp** | **-0.0667 (-7.41%)** | **+22.58 FPS** |

*For complete per-class breakdowns, confusion matrices, PR curves, and error analysis, see [`yolov8/results/FINAL_YOLO_EXPERIMENT_REPORT.md`](file:///c:/wildlife_COD/yolov8/results/FINAL_YOLO_EXPERIMENT_REPORT.md).*

---

## 4. Production API & Downstream Deep SORT Contract

### Python API Usage:
```python
import cv2
from yolov8.src.detect import WildlifeDetector

# Initialize detector with verified Model B checkpoint
detector = WildlifeDetector(
    model_path="yolov8/results/camouflage_yolov8s/weights/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.60
)

# Inference on video frame
frame = cv2.imread("sample_frame.jpg")
detections = detector.detect(frame)

# Structured output for Deep SORT tracker
print(detections)
```

### Output JSON Schema:
```json
[
  {
    "class_id": 0,
    "class_name": "buffalo",
    "confidence": 0.9412,
    "bbox": [145, 210, 480, 560]
  }
]
```

---

## 5. Verification & Test Suite

The test suite contains 17 automated tests covering detection accuracy, invalid input handling, model structure, and downstream contract compliance:

```bash
pytest
```
**Status: 17 passed, 0 failed, 0 skipped.**

---

## 6. Project Directory Layout

```text
wildlife_COD/
├── README.md                                # Root system documentation
├── pytest.ini                               # Pytest configuration
├── .gitignore                               # Git ignore rules (weights, data, caches)
└── yolov8/
    ├── README.md                            # Detailed YOLOv8 module documentation
    ├── config/                              # Dataset configurations
    │   ├── baseline.yaml                    # Model A baseline data config
    │   ├── camouflage.yaml                  # Model B camo-aware data config
    │   ├── camo_benchmark.yaml              # Held-out camo test config
    │   └── negatives_benchmark.yaml         # True negative test config
    ├── src/                                 # Production detection source code
    │   ├── detect.py                        # WildlifeDetector production API
    │   ├── inference.py                     # CLI & streaming inference runner
    │   ├── utils.py                         # Geometry, clipping & Deep SORT formatter
    │   └── ...                              # Data acquisition & audit scripts
    ├── tests/                               # Automated test suite (17 tests)
    │   ├── test_detection.py
    │   ├── test_invalid_input.py
    │   ├── test_model.py
    │   └── test_output_format.py
    └── results/                             # Consolidated experimental records
        ├── FINAL_YOLO_EXPERIMENT_REPORT.md  # Comprehensive experiment report
        ├── baseline_yolov8s/                # Frozen Model A weights & plots
        ├── camouflage_yolov8s/              # Frozen Model B weights & plots
        └── camouflage_yolov8s_error_analysis/ # Visual comparison panels
```
