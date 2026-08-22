"""
Production Inference Runner for YOLOv8 Wildlife Detection Subsystem

Supports:
1. Single image file inference
2. Image directory batch processing
3. Video file stream processing
4. Real-time webcam / RTSP stream processing

Keeps inference invocation logic clean and decoupled from core detector engine.
"""

from typing import List, Dict, Any, Optional
import os
import sys
import glob
import time
import argparse
import cv2
import numpy as np
from yolov8.src.detect import WildlifeDetector, DEFAULT_YOLOV8S_CHECKPOINT


def run_image_inference(
    detector: WildlifeDetector,
    image_path: str,
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute detection on a single image file.
    
    Args:
        detector: Initialized WildlifeDetector instance.
        image_path: Path to input image.
        output_path: Optional path to save visual detection output.
        
    Returns:
        List of structured detection dictionaries.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not decode image at: {image_path}")

    detections = detector.detect(frame)

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        annotated = detector.draw_detections(frame, detections)
        cv2.imwrite(output_path, annotated)

    return detections


def run_directory_inference(
    detector: WildlifeDetector,
    input_dir: str,
    output_dir: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Execute batch detection across an image directory.
    
    Args:
        detector: Initialized WildlifeDetector instance.
        input_dir: Directory containing image files.
        output_dir: Optional directory to save annotated visual outputs.
        
    Returns:
        Dictionary mapping image path to list of structured detections.
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Directory not found: {input_dir}")

    results: Dict[str, List[Dict[str, Any]]] = {}
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files: List[str] = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    image_files = sorted(list(set(image_files)))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
    
    Args:
        detector: Initialized WildlifeDetector instance.
        video_path: Path to video file.
        output_path: Optional path to save output annotated video.
        max_frames: Optional limit on number of frames to process.
        
    Returns:
        Summary metrics including latency, FPS, and per-frame detections.
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
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
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
            fps_current = 1.0 / max(latency, 1e-5)
            cv2.putText(
                annotated,
                f"FPS: {fps_current:.1f} | Dets: {len(dets)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
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
        "video_path": video_path,
        "frames_processed": frame_count,
        "total_time_sec": round(total_latency_sec, 4),
        "average_fps": round(avg_fps, 2),
        "average_latency_ms": round(avg_latency_ms, 2),
        "total_detections_count": sum(len(d) for d in all_detections),
        "detections": all_detections
    }


def run_webcam_inference(
    detector: WildlifeDetector,
    camera_id: int = 0,
    max_frames: Optional[int] = None,
    display: bool = False
) -> Dict[str, Any]:
    """
    Execute real-time streaming inference on webcam or RTSP camera stream.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not access camera source: {camera_id}")

    frame_count = 0
    total_time_sec = 0.0
    all_detections = []

    try:
        while True:
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
            total_time_sec += latency
            all_detections.append(dets)

            if display:
                annotated = detector.draw_detections(frame, dets)
                fps = 1.0 / max(latency, 1e-5)
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("YOLOv8s Wildlife Detection Stream", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    avg_fps = frame_count / max(total_time_sec, 1e-5) if frame_count > 0 else 0.0
    return {
        "frames_processed": frame_count,
        "average_fps": round(avg_fps, 2),
        "total_detections_count": sum(len(d) for d in all_detections)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 Wildlife Subsystem Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to image, directory, video file, or 'webcam'")
    parser.add_argument("--model", type=str, default=DEFAULT_YOLOV8S_CHECKPOINT, help="Model weights path")
    parser.add_argument("--output", type=str, default=None, help="Output image/video/directory path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="CUDA device index or 'cpu'")

    args = parser.parse_args()

    detector = WildlifeDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )

    if args.source.lower() == "webcam":
        run_webcam_inference(detector, camera_id=0, display=True)
    elif os.path.isdir(args.source):
        results = run_directory_inference(detector, args.source, args.output)
        print(f"Processed {len(results)} images in directory {args.source}")
    elif os.path.isfile(args.source):
        ext = os.path.splitext(args.source)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            summary = run_video_inference(detector, args.source, args.output)
            print(f"Processed {summary['frames_processed']} video frames at {summary['average_fps']} FPS")
        else:
            dets = run_image_inference(detector, args.source, args.output)
            print(f"Detected {len(dets)} wildlife objects in {args.source}: {dets}")
