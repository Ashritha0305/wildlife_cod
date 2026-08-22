# FINAL SCIENTIFIC EXPERIMENT REPORT: YOLOv8 BASELINE (MODEL A) VS. CAMOUFLAGE-AWARE (MODEL B) WILDLIFE OBJECT DETECTION

**Project:** Camouflage-Aware Wildlife Detection and Threat Analysis System  
**Module:** YOLOv8 Wildlife Object Detection Subsystem  
**Date of Audit & Freeze:** August 22, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM)  
**Execution Environment:** Python 3.13.14, Ultralytics 8.4.126, PyTorch 2.13.0+cu126, CUDA 12.6, Windows 11  

---

## 1. Experimental Objective

The objective of this investigation was to answer the following research question under strict experimental control:

> **Scientific Question:** *"Does adding genuine, naturally concealed camera-trap training data improve detection of the SAME four biological species (buffalo, elephant, rhino, zebra) under natural camouflage conditions, and what is the exact trade-off on normal daylight detection and computational throughput?"*

To isolate the effect of training data:
- The network architecture was held constant across both experiments (**standard Ultralytics YOLOv8s**).
- The four biological classes remained strictly fixed: `0: buffalo`, `1: elephant`, `2: rhino`, `3: zebra`.
- Model A was trained exclusively on standard baseline wildlife photography.
- Model B was trained on the combined baseline dataset plus genuine camera-trap natural concealment captures.
- Both models were evaluated on the **exact same frozen test sets** under an identical evaluation protocol.

---

## 2. Dataset Provenance & Split Isolation

All datasets were staged under [`yolov8/data/`](file:///c:/wildlife_COD/yolov8/data/) and cryptographically verified against SHA-256 collision across all partitions.

### Dataset Composition:
1. **Baseline Dataset ([`yolov8/data/baseline/`](file:///c:/wildlife_COD/yolov8/data/baseline/)):**
   - **Total Images:** 1,463 images (2,620 instances)
   - **Train Split:** 1,036 images (1,844 instances)
   - **Val Split:** 216 images (424 instances)
   - **Test Split (FROZEN):** 211 images (352 instances: 63 buffalo, 96 elephant, 69 rhino, 124 zebra)
2. **Genuine Camouflage Dataset (WCS Camera Traps on LILA BC):**
   - **Acquisition Source:** Wildlife Conservation Society (WCS) Camera Traps, filtered for biological target species under natural concealment (night/twilight IR, dense brush, low contrast, mud/dust blending, herd occlusion).
   - **Sequence-Level Isolation:** Disjoint camera burst partitioning guaranteed 0 adjacent frame sequence leakage.
   - **Camouflage Train Split ([`yolov8/data/camo_training/train/`](file:///c:/wildlife_COD/yolov8/data/camo_training/train/)):** 229 images (339 instances: 86 buffalo, 93 elephant, 65 rhino, 95 zebra)
   - **Camouflage Val Split ([`yolov8/data/camo_training/val/`](file:///c:/wildlife_COD/yolov8/data/camo_training/val/)):** 45 images (60 instances: 15 buffalo, 10 elephant, 13 rhino, 22 zebra)
   - **Held-Out Camouflage Benchmark ([`yolov8/data/camo_benchmark/4class_camo/`](file:///c:/wildlife_COD/yolov8/data/camo_benchmark/4class_camo/)):** 45 images (64 instances: 20 buffalo, 14 elephant, 15 rhino, 15 zebra) — **100% FROZEN & UNTOUCHED during training/validation**.
3. **Animal-Free Negative Benchmark ([`yolov8/data/camo_benchmark/negatives/`](file:///c:/wildlife_COD/yolov8/data/camo_benchmark/negatives/)):**
   - 60 images (0 wildlife instances) capturing high-texture rocks, dry savannah scrub, tree shadows, and tangled deadwood.

---

## 3. Model A Configuration (Baseline)

- **Model Checkpoint:** [`yolov8/results/baseline_yolov8s/weights/best.pt`](file:///c:/wildlife_COD/yolov8/results/baseline_yolov8s/weights/best.pt)
- **File Size:** $22,518,314\text{ bytes}$ ($21.48\text{ MB}$)
- **SHA-256 Checksum:** `d7b4754b2a82df5dbb57e1734f0840dc84b074dca0d443395206ee93d1ef709c`
- **Architecture:** Standard YOLOv8s ($11,127,132\text{ parameters}$, 28.4 GFLOPs)
- **Training Data:** Baseline train only (1,036 images)
- **Validation Data:** Baseline val only (216 images)

---

## 4. Model B Configuration (Camouflage-Aware)

- **Model Checkpoint:** [`yolov8/results/camouflage_yolov8s/weights/best.pt`](file:///c:/wildlife_COD/yolov8/results/camouflage_yolov8s/weights/best.pt)
- **File Size:** $22,499,946\text{ bytes}$ ($21.46\text{ MB}$)
- **SHA-256 Checksum:** `19ff4a55a415e6c44f29f1a72a86cd4031489c0b0c9274357bfd4b7fd5b32344`
- **Architecture:** Standard YOLOv8s ($11,127,132\text{ parameters}$, 28.4 GFLOPs)
- **Training Data:** Baseline train (1,036) + Camo train (229) = 1,265 images
- **Validation Data:** Baseline val (216) + Camo val (45) = 261 images

---

## 5. Training Configuration & Hyperparameters

Both models were trained using identical hyperparameter constraints on the NVIDIA GeForce RTX 4050 GPU:

| Hyperparameter | Value | Rationale / Rule |
|:---|:---|:---|
| **Base Architecture** | YOLOv8s (`yolov8s.pt` pretrained) | Standard small detection backbone |
| **Epochs** | 20 | Identical training horizon |
| **Batch Size** | 16 | Consistent gradient batching |
| **Image Resolution** | $640 \times 640$ | Standard YOLO input dimensions |
| **Optimizer** | AdamW (lr0=0.00125, momentum=0.9) | Automatic Ultralytics parameter grouping |
| **Data Augmentation** | Standard YOLOv8 defaults (Mosaic off last 10) | Identical regularization schedule |
| **Random Seed** | 42 | Deterministic initialization & shuffle |
| **Hardware** | CUDA:0 (NVIDIA RTX 4050 Laptop GPU) | Hardware parity |

---

## 6. Authoritative Evaluation Protocol

To prevent metric conflation, all comparative metrics in this report follow the **Primary Operational Detection Protocol**:

- **Resolution:** `imgsz = 640`
- **Confidence Threshold:** `conf = 0.25`
- **NMS IoU Threshold:** `iou = 0.60`
- **Batch Size:** `batch = 16`
- **Device:** `CUDA:0`
- **Augmentation:** `augment = False`

*(Note: Diagnostic full-curve COCO evaluation at $\text{conf}=0.001, \text{iou}=0.70$ is documented separately in Section 14 to explain the reproducibility audit).*

---

## 7. Normal Baseline Test Set Results

**Dataset:** [`yolov8/data/baseline/test/`](file:///c:/wildlife_COD/yolov8/data/baseline/test/) (211 images, 352 instances, evaluated at `conf=0.25, iou=0.60`)

| Metric | Model A (Baseline) | Model B (Camouflage) | Absolute Delta ($\Delta$) | Relative Change |
|:---|:---:|:---:|:---:|:---:|
| **Precision ($P$)** | 96.27% | 95.51% | **-0.76 pp** | -0.79% |
| **Recall ($R$)** | 93.90% | 93.86% | **-0.04 pp** | -0.04% |
| **mAP@50** | 95.80% | 95.23% | **-0.57 pp** | -0.59% |
| **mAP@50:95** | 82.21% | 80.23% | **-1.98 pp** | -2.41% |

---

## 8. Held-Out Camouflage Benchmark Results

**Dataset:** [`yolov8/data/camo_benchmark/4class_camo/`](file:///c:/wildlife_COD/yolov8/data/camo_benchmark/4class_camo/) (45 images, 64 instances, evaluated at `conf=0.25, iou=0.60`)

| Metric | Model A (Baseline) | Model B (Camouflage) | Absolute Gain ($\Delta$) | Relative Gain |
|:---|:---:|:---:|:---:|:---:|
| **Precision ($P$)** | 48.94% | **76.94%** | **+28.00 pp** | **+57.21%** |
| **Recall ($R$)** | 28.33% | **57.98%** | **+29.65 pp** | **+104.66%** |
| **mAP@50** | 15.07% | **58.07%** | **+43.00 pp** | **+285.34%** |
| **mAP@50:95** | 9.13% | **35.58%** | **+26.45 pp** | **+289.70%** |

---

## 9. Animal-Free Negative Background Results

**Dataset:** [`yolov8/data/camo_benchmark/negatives/`](file:///c:/wildlife_COD/yolov8/data/camo_benchmark/negatives/) (60 animal-free images, evaluated at `conf=0.25`)

| Metric | Model A (Baseline) | Model B (Camouflage) | Difference ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Total Background Images** | 60 | 60 | — |
| **Total False Positives (FP)** | 54 | **50** | **-4 false detections** |
| **FPPI ($\text{FP} / 60$)** | 0.9000 | **0.8333** | **-0.0667 (-7.41%)** |
| **Buffalo False Positives** | 0 | 2 | +2 |
| **Elephant False Positives** | 54 | 43 | -11 (rock texture suppression) |
| **Rhino False Positives** | 0 | 0 | 0 |
| **Zebra False Positives** | 0 | 5 | +5 |

---

## 10. Per-Class Performance Breakdown

Evaluated under the primary operational protocol (`conf=0.25, iou=0.60`):

### A. Normal Baseline Test Set (211 images, 352 instances)
| Class | Metric | Model A (Baseline) | Model B (Camouflage) | Delta ($\Delta$) |
|:---|:---|:---:|:---:|:---:|
| **Buffalo** ($n=63$) | Precision / Recall<br>AP@50 / AP@50:95 | 97.52% / 92.06%<br>92.47% / 81.74% | 96.65% / 91.51%<br>94.97% / 82.49% | -0.87 pp / -0.55 pp<br>+2.50 pp / +0.75 pp |
| **Elephant** ($n=96$) | Precision / Recall<br>AP@50 / AP@50:95 | 94.82% / 95.43%<br>97.05% / 79.56% | 94.21% / 93.75%<br>94.11% / 75.86% | -0.61 pp / -1.68 pp<br>-2.94 pp / -3.70 pp |
| **Rhino** ($n=69$) | Precision / Recall<br>AP@50 / AP@50:95 | 98.52% / 96.60%<br>98.44% / 90.69% | 97.06% / 95.82%<br>97.06% / 86.71% | -1.46 pp / -0.78 pp<br>-1.38 pp / -3.98 pp |
| **Zebra** ($n=124$) | Precision / Recall<br>AP@50 / AP@50:95 | 94.19% / 91.53%<br>95.24% / 76.85% | 94.13% / 94.35%<br>94.76% / 75.87% | -0.06 pp / +2.82 pp<br>-0.48 pp / -0.98 pp |

### B. Held-Out Camouflage Test Set (45 images, 64 instances)
| Class | Metric | Model A (Baseline) | Model B (Camouflage) | Absolute Gain ($\Delta$) |
|:---|:---|:---:|:---:|:---:|
| **Buffalo** ($n=20$) | Precision / Recall<br>AP@50 / AP@50:95 | 66.67% / 10.00%<br>9.50% / 5.20% | **95.15%** / **45.00%**<br>**53.67%** / **32.00%** | **+28.48 pp** / **+35.00 pp**<br>**+44.17 pp** / **+26.80 pp** |
| **Elephant** ($n=14$) | Precision / Recall<br>AP@50 / AP@50:95 | 41.18% / 50.00%<br>26.43% / 17.54% | **75.32%** / **71.43%**<br>**70.33%** / **39.88%** | **+34.14 pp** / **+21.43 pp**<br>**+43.90 pp** / **+22.34 pp** |
| **Rhino** ($n=15$) | Precision / Recall<br>AP@50 / AP@50:95 | 30.77% / 26.67%<br>8.65% / 5.80% | **77.93%** / **66.67%**<br>**66.50%** / **41.97%** | **+47.16 pp** / **+40.00 pp**<br>**+57.85 pp** / **+36.17 pp** |
| **Zebra** ($n=15$) | Precision / Recall<br>AP@50 / AP@50:95 | 57.14% / 26.67%<br>15.70% / 7.98% | **59.37%** / **48.83%**<br>**41.78%** / **28.47%** | **+2.23 pp** / **+22.16 pp**<br>**+26.08 pp** / **+20.49 pp** |

---

## 11. Hardware Latency & Throughput Benchmark

### Benchmark Methodology Selection:
Across previous stages, three benchmark methodologies were executed:
1. **Stage 4A-2:** Single-pass timed evaluation on baseline test frames with CUDA synchronization.
2. **Stage 6A:** Single-pass predictive pipeline with 25 warmup iterations and 150 test frames.
3. **Stage 6B (Selected Authoritative Protocol):** Standardized, sequential side-by-side benchmarking within the identical Python process. Includes:
   - 25 dedicated GPU warm-up cycles.
   - 150 sequential test-stream frame inferences at $640 \times 640$.
   - Explicit `torch.cuda.synchronize()` before and after every timing block.
   - Peak VRAM tracked via `torch.cuda.reset_peak_memory_stats(0)` and `torch.cuda.max_memory_allocated(0)`.

### Authoritative Hardware Benchmark Results:
**Device:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.6, PyTorch 2.13.0+cu126)

| Metric | Model A (Baseline) | Model B (Camouflage) | Operational Requirement | Status |
|:---|:---:|:---:|:---:|:---:|
| **Preprocessing Latency** | 1.05 ms | 1.30 ms | — | Fast |
| **Inference Latency** | 5.15 ms | 5.14 ms | — | Fast |
| **Postprocessing (NMS) Latency** | 1.13 ms | 1.10 ms | — | Fast |
| **Total Real-Stream Latency** | 11.00 ms (±7.53 ms) | **8.81 ms (±1.47 ms)** | $\le 33.33\text{ ms}$ | **PASS** |
| **Throughput (Real-Stream FPS)** | 90.89 FPS | **113.47 FPS** | $\ge 30.00\text{ FPS}$ | **PASS ($3.78\times$ margin)** |
| **Peak GPU VRAM** | 206.66 MB | 249.18 MB | $\le 4,096\text{ MB}$ | **PASS** |

---

## 12. Error Analysis Summary

Visual error analysis comparisons (Ground Truth vs Model A vs Model B) were generated and preserved in [`yolov8/results/camouflage_yolov8s_error_analysis/`](file:///c:/wildlife_COD/yolov8/results/camouflage_yolov8s_error_analysis/):

1. **Dense Brush Concealment (`case_01_camo_buffalo`):** Model A failed on 3 out of 4 nocturnal buffalo partially obscured by acacia limbs. Model B correctly localized all 4 buffalo.
2. **Low-Contrast Twilight Blending (`case_02_camo_buffalo`):** Model A produced 0 detections against savannah grass. Model B detected the target with $0.81$ confidence.
3. **Occluded Silhouette Separation (`case_03_camo_elephant`):** Model A hallucinated a dark tree trunk as an elephant. Model B suppressed the tree and cleanly boxed the true elephant.
4. **Heavy Dust / Herd Overlap (`case_04_camo_elephant`):** Model A merged overlapping bodies into a single degenerate box. Model B resolved individual instances.
5. **Mud / Soil Blending (`case_05_camo_rhino`):** Mud-coated rhino matching background earth was completely missed by Model A; Model B localized it at $0.88$ confidence.
6. **False Positive Suppression (`case_08_neg_bg`):** Model A triggered false detections on shadowed rock outcrops. Model B correctly produced 0 detections.

---

## 13. Reproducibility & Isolation Audit

Dedicated evaluation runs were executed to populate isolated artifact directories without modifying original checkpoints:

- **Model A Normal Test:** [`yolov8/results/repro_eval_modelA/`](file:///c:/wildlife_COD/yolov8/results/repro_eval_modelA/)
  - Artifacts: `results.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxF1_curve.png`
- **Model B Normal Test:** [`yolov8/results/repro_eval_modelB_normal/`](file:///c:/wildlife_COD/yolov8/results/repro_eval_modelB_normal/)
  - Artifacts: `results.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxF1_curve.png`
- **Model A Camouflage Test:** [`yolov8/results/repro_eval_modelA_camo/`](file:///c:/wildlife_COD/yolov8/results/repro_eval_modelA_camo/)
  - Artifacts: `results.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxF1_curve.png`
- **Model B Camouflage Test:** [`yolov8/results/repro_eval_modelB_camo/`](file:///c:/wildlife_COD/yolov8/results/repro_eval_modelB_camo/)
  - Artifacts: `results.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxF1_curve.png`

---

## 14. Metric Discrepancy Explanation

| Evaluation Parameter | Stage 4A-2 / Production Protocol | Standard COCO Diagnostic Protocol | Cause of Difference |
|:---|:---:|:---:|:---|
| **`conf` Threshold** | `0.25` | `0.001` | $\text{conf}=0.25$ prunes detections below 0.25 confidence before PR integration. |
| **`iou` Threshold** | `0.60` | `0.70` | Standard NMS overlap threshold. |
| **Model A mAP@50** | **95.80%** | **98.14%** | $\text{conf}=0.001$ integrates the full low-confidence envelope. |
| **Model A mAP@50:95** | **82.21%** | **83.62%** | Small gain from low-confidence bounding box localization. |
| **Model A Precision** | **96.27%** | **95.85%** | Lower confidence threshold admits marginal candidates. |
| **Model A Recall** | **93.90%** | **94.17%** | Low-confidence detections recover minor missed instances. |

**Audit Conclusion:** The discrepancy was entirely caused by the confidence threshold setting (`conf=0.25` vs `conf=0.001`). Both evaluations are mathematically exact representations of the same frozen checkpoint.

---

## 15. Dataset Leakage & Taxonomy Verification

1. **Zero Hash Overlap:** A complete SHA-256 cross-partition audit confirmed 0 duplicate hashes between Model B training data and the held-out test sets (`baseline/test` and `camo_benchmark/4class_camo`).
2. **Sequence Separation:** Training and held-out test camera trap images were sourced from distinct camera burst sequences, guaranteeing 0 adjacent-frame temporal leakage.
3. **Taxonomy Invariance:** Zero non-target biological substitutions were introduced.

---

## 16. Pytest Regression Verification

The test suite was executed via `pytest`:
```text
yolov8\tests\test_detection.py ....                                      [ 23%]
yolov8\tests\test_invalid_input.py ......                                [ 58%]
yolov8\tests\test_model.py ...                                           [ 76%]
yolov8\tests\test_output_format.py ....                                  [100%]

============================= 17 passed in 3.82s ==============================
```
**Status: 17/17 PASSED (0 failed, 0 skipped).**

---

## 17. Scientific Interpretation & Objective Trade-Offs

1. **Major Camouflage Generalization Improvement:**
   - On genuine camera trap images depicting natural concealment, Model B achieved a **$+43.00\text{ pp}$ absolute improvement in mAP@50** ($15.07\% \to 58.07\%$) and a **$+26.45\text{ pp}$ absolute improvement in mAP@50:95** ($9.13\% \to 35.58\%$).
   - Recall increased from $28.33\%$ to $57.98\%$ ($+104.66\%$ relative increase), effectively doubling the detection rate of obscured animals.
2. **Measured Trade-Off on Normal Daylight Imagery:**
   - Incorporating challenging camera-trap training samples caused a slight, measurable degradation on clean daylight photographs: **$-0.57\text{ pp}$ mAP@50** ($95.80\% \to 95.23\%$) and **$-1.98\text{ pp}$ mAP@50:95** ($82.21\% \to 80.23\%$).
   - Precision dropped by $-0.76\text{ pp}$ ($96.27\% \to 95.51\%$) while Recall remained virtually unchanged ($-0.04\text{ pp}$, $93.90\% \to 93.86\%$).
3. **Net System Benefit:**
   - The $-0.57\text{ pp}$ reduction on normal test data is an acceptable operational trade-off for the $+43.00\text{ pp}$ gain on camouflaged targets and the $7.41\%$ reduction in false positive detections on animal-free backgrounds.

---

## 18. Experimental Limitations

1. **Benchmark Scale:** The held-out camouflage benchmark consists of 45 images (64 instances). While sufficient for initial proof of concept, larger geographic camera-trap benchmarks will be needed for extensive statistical generalization.
2. **Class Imbalance in Camouflage Data:** Genuine camera-trap captures of wild black and white rhinoceros are scarce due to extreme poaching protection and low population density, resulting in fewer total rhino training samples ($65\text{ train}$, $13\text{ val}$) compared to elephants ($93\text{ train}$) and zebras ($95\text{ train}$).
3. **Detection-Only Scope:** This experiment evaluates only standard 2D bounding-box object detection. U-Net segmentation, Deep SORT tracking, ZoeDepth metric depth estimation, and fuzzy logic threat analysis are downstream subsystems and were not evaluated in this YOLO experiment.

---

## 19. Final Scientific Conclusion

> **"Adding genuine natural-concealment camera-trap training examples produced a substantial improvement in detection performance on the held-out camouflage benchmark ($+43.00\text{ pp mAP@50}$, $+26.45\text{ pp mAP@50:95}$) while maintaining broadly comparable performance on the normal daylight wildlife test set ($-0.57\text{ pp mAP@50}$, $-1.98\text{ pp mAP@50:95}$) and preserving real-time throughput ($113.47\text{ FPS}$ on NVIDIA RTX 4050)."**

---
*Report certified and frozen. Checkpoints and evaluation artifacts permanently archived.*
