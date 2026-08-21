# Camouflage-Aware Wildlife Detection and Threat Analysis System

## Module: YOLOv8 Wildlife Object Detection Subsystem

### 1. Final Project Architecture
This module implements the **YOLOv8 wildlife object detection engine** within the multi-stage surveillance and threat analysis pipeline:

```text
Camera / Video Stream
       │
   ┌───┴───┐
   ↓       ↓
 U-Net   Optical Flow
 (Camo)  (Motion)
   └───┬───┘
       ↓
     YOLOv8 Wildlife Object Detection [THIS MODULE]
       ↓
   Deep SORT Multi-Object Tracking
       ↓
   ZoeDepth Depth / Distance Estimation
       ↓
   Rule-Based + Fuzzy Logic Threat Analysis
       ↓
   Threat Alert / Visualization
```

---

### 2. Module Responsibilities & System Boundaries

#### Scope of YOLOv8 Subsystem:
- Localize wildlife objects in real-time image and video frames.
- Predict bounding-box coordinates (`[x1, y1, x2, y2]` in pixel units).
- Predict species-level class identity (`tiger`, `elephant`, `deer`, `leopard`, etc.) and class IDs.
- Provide structured detection dictionaries directly consumable by Deep SORT.

#### Downstream Subsystems (Handled by Teammates' Modules):
- **Multi-Object Tracking:** Handled exclusively by Deep SORT (assigns persistent track IDs and motion trajectory).
- **Depth / Distance Estimation:** Handled exclusively by ZoeDepth (estimates metric depth and physical proximity).
- **Threat Assessment:** Handled exclusively by Rule-Based & Fuzzy Logic subsystem (calculates threat levels from species, distance, velocity, and camouflage status).

---

### 3. Downstream Interface Contract (Deep SORT Integration)
The detection subsystem outputs structured detection objects for each frame:

```json
[
  {
    "class_id": 1,
    "class_name": "tiger",
    "confidence": 0.94,
    "bbox": [320, 180, 670, 520]
  }
]
```

- `bbox`: Absolute pixel coordinates `[x1, y1, x2, y2]` corresponding to `[xmin, ymin, xmax, ymax]`.
- `confidence`: Calibrated float score $\in [0.0, 1.0]$.
- `class_id`: Non-negative integer mapped to the dataset class YAML.
- `class_name`: True biological species name.

---

### 4. Current Status
* **Stage:** Stage 2 Complete (Environment, PyTorch CUDA 12.6, RTX 4050 GPU, Ultralytics Verified).
* **Dataset State:** Stage 3 Strategy Analysis (Candidate Evaluation & Species-Level Strategy Definition).
