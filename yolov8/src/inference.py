"""
Inference Runner for YOLOv8 Wildlife Detection Subsystem
Supports:
1. Single image file
2. Image directory batch processing
3. Video file stream processing
4. Real-time webcam / RTSP stream
"""

from typing import List, Dict, Any, Optional
import os
import sys
import glob
import time
import cv2
import numpy as np
from yolov8.src.detect import WildlifeDetector


def run_image_inference(
    detector: WildlifeDetector,
    image_path: str,
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Execute detection on a single image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not decode image at: {image_path}")

    detections = detector.detect(frame)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        annotated = detector.draw_detections(frame, detections)
        cv2.imwrite(output_path, annotated)

    return detections


def run_directory_inference(
    detector: WildlifeDetector,
    input_dir: str,
    output_dir: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Execute batch detection across an image directory."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Directory not found: {input_dir}")

    results = {}
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    for img_p in image_files:
        base = os.path.basename(img_p)
        out_p = os.path.join(output_dir, f"det_{base}") if output_dir else None
        dets = run_image_inference(detector, img_p, out_p)
        results[img_p] = dets

    return results


def run_video_inference(
    detector: WildlifeDetector,
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute frame-by-frame inference on a video file.
    Measures elapsed time, processed frames, and average FPS.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    frame_count = 0
    total_latency_sec = 0.0
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()

        latency = t1 - t0
        total_latency_sec += latency
        all_detections.append(dets)

        if writer:
            annotated = detector.draw_detections(frame, dets)
            # Overlay current FPS
            fps_current = 1.0 / max(latency, 1e-5)
            cv2.putText(
                annotated,
                f"FPS: {fps_current:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()

    avg_fps = frame_count / max(total_latency_sec, 1e-5) if frame_count > 0 else 0.0
    avg_latency_ms = (total_latency_sec / max(frame_count, 1)) * 1000.0

    return {
        "frames_processed": frame_count,
        "total_time_sec": round(total_latency_sec, 4),
        "average_fps": round(avg_fps, 2),
        "average_latency_ms": round(avg_latency_ms, 2),
        "detections": all_detections
    }
