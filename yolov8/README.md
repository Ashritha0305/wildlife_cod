# YOLOv8 Wildlife Object Detection Subsystem

**Module:** YOLOv8 Wildlife Object Detection Subsystem  
**Parent System:** Camouflage-Aware Wildlife Detection and Threat Analysis System  
**Selected Architecture:** Ultralytics YOLOv8s (Small) — 11.13 Million Parameters, 28.4 GFLOPs  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM, CUDA 12.6)  
**Supported Target Classes:** `0: buffalo`, `1: elephant`, `2: rhino`, `3: zebra`  

---

## 1. Subsystem Architecture & Responsibilities

The YOLOv8 Wildlife Object Detection module operates as a high-throughput, standalone 2D object detection engine within the multi-stage surveillance and threat analysis system:

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

### Module Responsibilities:
- Receive single image frames or continuous video streams (BGR format).
- Perform high-speed bounding-box localization, species classification, and confidence scoring.
- Enforce strict coordinate boundaries and spatial validation ($0 \le x_1 < x_2 \le \text{width}$, $0 \le y_1 < y_2 \le \text{height}$).
- Output deterministic, structured detection dictionaries conforming to the downstream Deep SORT interface contract.

### Explicit Subsystem Boundaries (Out of Scope for YOLO):
- **Multi-Object Tracking / Track IDs:** Handled exclusively by Deep SORT (Kalman filtering + ReID association).
- **Metric Depth / Distance Estimation:** Handled exclusively by ZoeDepth.
- **Threat Level Scoring & Fuzzy Logic:** Handled by the Rule-Based & Fuzzy Logic subsystem.
- **Emergency Notifications / Alerts:** Handled by the Alert & Notification subsystem.
- **No Neural Feature Fusion:** YOLOv8 remains a standard, self-contained CNN architecture.

---

## 2. Models & Checkpoints

| Model | Checkpoint Path | SHA-256 Checksum | Training Data | Description |
|:---|:---|:---:|:---|:---|
| **Model A (Baseline)** | [`yolov8/results/baseline_yolov8s/weights/best.pt`](file:///c:/wildlife_COD/yolov8/results/baseline_yolov8s/weights/best.pt) | `d7b4754b...` | 1,036 baseline train | Trained exclusively on clean daylight wildlife photography. |
| **Model B (Camouflage)** | [`yolov8/results/camouflage_yolov8s/weights/best.pt`](file:///c:/wildlife_COD/yolov8/results/camouflage_yolov8s/weights/best.pt) | `19ff4a55...` | 1,036 baseline + 229 genuine camo | Trained on baseline plus camera-trap natural concealment captures. |

---

## 3. Dataset Architecture & Split Integrity

All datasets reside under [`yolov8/data/`](file:///c:/wildlife_COD/yolov8/data/) and are verified for 0 cross-partition hash collision and 0 sequence burst leakage:

- **Baseline Dataset (`yolov8/data/baseline/`):**
  - `train`: 1,036 images (1,844 instances)
  - `val`: 216 images (424 instances)
  - `test` (FROZEN): 211 images (352 instances: 63 buffalo, 96 elephant, 69 rhino, 124 zebra)
- **Camouflage Training Dataset (`yolov8/data/camo_training/`):**
  - `train`: 229 images (339 instances)
  - `val`: 45 images (60 instances)
- **Held-Out Camouflage Benchmark (`yolov8/data/camo_benchmark/4class_camo/`):**
  - `test` (FROZEN): 45 images (64 instances: 20 buffalo, 14 elephant, 15 rhino, 15 zebra)
- **Animal-Free Negative Benchmark (`yolov8/data/camo_benchmark/negatives/`):**
  - 60 images (0 wildlife instances) capturing dry brush, rocky terrain, tree shadows.

---

## 4. Authoritative Experimental Results

**Primary Operational Evaluation Protocol:** `imgsz = 640`, `conf = 0.25`, `iou = 0.60`, `batch = 16`, `device = CUDA:0`, `augment = False`.

### Overall Benchmark Performance
| Evaluation Split | Metric | Model A (Baseline) | Model B (Camouflage) | Delta ($\Delta$) | Relative Change |
|:---|:---|:---:|:---:|:---:|:---:|
| **Normal Test Set**<br>(211 images, 352 instances) | Precision<br>Recall<br>mAP@50<br>mAP@50:95 | 96.27%<br>93.90%<br>95.80%<br>82.21% | 95.51%<br>93.86%<br>95.23%<br>80.23% | -0.76 pp<br>-0.04 pp<br>-0.57 pp<br>-1.98 pp | -0.79%<br>-0.04%<br>-0.59%<br>-2.41% |
| **Held-Out Camouflage Benchmark**<br>(45 images, 64 instances) | Precision<br>Recall<br>mAP@50<br>mAP@50:95 | 48.94%<br>28.33%<br>15.07%<br>9.13% | **76.94%**<br>**57.98%**<br>**58.07%**<br>**35.58%** | **+28.00 pp**<br>**+29.65 pp**<br>**+43.00 pp**<br>**+26.45 pp** | **+57.21%**<br>**+104.66%**<br>**+285.34%**<br>**+289.70%** |
| **Negative Background Benchmark**<br>(60 animal-free images) | False Positives (FP)<br>FPPI ($\text{FP} / 60$) | 54 FPs<br>0.9000 | **50 FPs**<br>**0.8333** | **-4 FPs**<br>**-0.0667** | **-7.41% FPPI reduction** |

### Per-Class Detailed Results (`conf=0.25, iou=0.60`)
| Split | Class | Model A Precision / Recall | Model B Precision / Recall | Model A AP@50 / AP@50:95 | Model B AP@50 / AP@50:95 |
|:---|:---|:---:|:---:|:---:|:---:|
| **Normal Test** | **Buffalo**<br>**Elephant**<br>**Rhino**<br>**Zebra** | 97.52% / 92.06%<br>94.82% / 95.43%<br>98.52% / 96.60%<br>94.19% / 91.53% | 96.65% / 91.51%<br>94.21% / 93.75%<br>97.06% / 95.82%<br>94.13% / 94.35% | 92.47% / 81.74%<br>97.05% / 79.56%<br>98.44% / 90.69%<br>95.24% / 76.85% | 94.97% / 82.49%<br>94.11% / 75.86%<br>97.06% / 86.71%<br>94.76% / 75.87% |
| **Camouflage Test** | **Buffalo**<br>**Elephant**<br>**Rhino**<br>**Zebra** | 66.67% / 10.00%<br>41.18% / 50.00%<br>30.77% / 26.67%<br>57.14% / 26.67% | **95.15%** / **45.00%**<br>**75.32%** / **71.43%**<br>**77.93%** / **66.67%**<br>**59.37%** / **48.83%** | 9.50% / 5.20%<br>26.43% / 17.54%<br>8.65% / 5.80%<br>15.70% / 7.98% | **53.67%** / **32.00%**<br>**70.33%** / **39.88%**<br>**66.50%** / **41.97%**<br>**41.78%** / **28.47%** |

---

## 5. Hardware Latency & Throughput Benchmark

**Device:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.6, PyTorch 2.13.0+cu126)  
**Standard Benchmark Protocol:** 25 warm-up cycles, 150 test-stream frames, CUDA synchronization before/after every timing block, peak VRAM memory tracking.

| Hardware Metric | Model A (Baseline) | Model B (Camouflage) | Target Threshold | Operational Status |
|:---|:---:|:---:|:---:|:---:|
| **Total Real-Stream Latency** | 11.00 ms (±7.53 ms) | **8.81 ms (±1.47 ms)** | $\le 33.33\text{ ms}$ | **PASS** |
| **Throughput (Real-Stream FPS)** | 90.89 FPS | **113.47 FPS** | $\ge 30.00\text{ FPS}$ | **PASS ($3.78\times$ margin)** |
| **Peak GPU VRAM** | 206.66 MB | 249.18 MB | $\le 4,096\text{ MB}$ | **PASS** |

---

## 6. Production Detection API & Deep SORT Integration

The production API is implemented in [`yolov8/src/detect.py`](file:///c:/wildlife_COD/yolov8/src/detect.py).

### Quick Usage:
```python
from yolov8.src.detect import WildlifeDetector
import cv2

# Initialize detector with calibrated weights
detector = WildlifeDetector(
    model_path="yolov8/results/camouflage_yolov8s/weights/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.60
)

# Detect on image / video frame
frame = cv2.imread("sample_frame.jpg")
detections = detector.detect(frame)

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

## 7. Pytest Verification Suite

The complete unit and integration regression test suite is located in [`yolov8/tests/`](file:///c:/wildlife_COD/yolov8/tests/):

```bash
pytest
```
- **Tests Executed:** 17
- **Passed:** **17**
- **Failed:** **0**
- **Skipped:** **0**

---

## 8. Experimental Limitations & Scope

1. **Benchmark Scale:** The held-out camouflage benchmark contains 45 images and 64 ground-truth instances.
2. **Species Distribution:** Rhinoceros camera-trap samples under natural concealment are scarce due to real-world rarity.
3. **Operational Trade-Off:** Adding challenging camera-trap training examples introduces a modest trade-off on normal daylight photography ($-0.57\text{ pp mAP@50}$ and $-1.98\text{ pp mAP@50:95}$).
4. **Subsystem Isolation:** This milestone certifies the standalone YOLOv8 object detector only. Downstream modules (U-Net, Deep SORT, ZoeDepth, Fuzzy Logic) are handled by separate subsystems.
