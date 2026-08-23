"""Command-line runner for end-to-end wildlife inference validation."""

import argparse
import json
import os
from typing import Any, Dict, List

import cv2

from yolov8.src.depth import DepthEstimator
from yolov8.src.detect import WildlifeDetector
from yolov8.src.e2e import EndToEndInference
from yolov8.src.threat import ThreatAnalyzer
from yolov8.src.tracking import ByteTrackTracker


def build_inference(args: argparse.Namespace) -> EndToEndInference:
    return EndToEndInference(
        detector=WildlifeDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        ),
        tracker=ByteTrackTracker(),
        depth_estimator=DepthEstimator(
            device=args.device if args.device in {"cpu", "cuda", "cuda:0"} else "cpu",
            backend=args.depth_backend,
            allow_fallback=args.depth_backend != "zoedepth",
        ),
        threat_analyzer=ThreatAnalyzer(
            warning_distance=args.warning_distance,
            critical_distance=args.critical_distance,
        ),
    )


def print_results(results: List[Dict[str, Any]]) -> None:
    print(json.dumps(results, indent=2))


def run_image(inference: EndToEndInference, source: str, output: str | None) -> None:
    frame = cv2.imread(source)
    if frame is None:
        raise ValueError(f"Could not decode image: {source}")
    results = inference.process_frame(frame, frame_id=1)
    if output:
        output_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(output_dir, exist_ok=True)
        rendered = inference.render_results(frame, results)
        if not cv2.imwrite(output, rendered):
            raise OSError(f"Could not write output image: {output}")
    print_results(results)


def run_video(inference: EndToEndInference, source: str, output: str | None) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    writer = None
    frame_id = 0
    try:
        if output:
            output_dir = os.path.dirname(os.path.abspath(output))
            os.makedirs(output_dir, exist_ok=True)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(
                output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise OSError(f"Could not write output video: {output}")
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_id += 1
            results = inference.process_frame(frame, frame_id=frame_id)
            print(f"Frame {frame_id}:")
            print_results(results)
            if writer:
                writer.write(inference.render_results(frame, results))
    finally:
        capture.release()
        if writer:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to an image or video")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO checkpoint path")
    parser.add_argument("--output", default=None, help="Annotated image or video output path")
    parser.add_argument("--device", default="cpu", help="YOLO/depth device: cpu, cuda, or cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--depth-backend", choices=("fallback", "auto", "zoedepth"), default="fallback")
    parser.add_argument("--warning-distance", type=float, default=20.0)
    parser.add_argument("--critical-distance", type=float, default=10.0)
    args = parser.parse_args()
    if not os.path.isfile(args.source):
        raise FileNotFoundError(f"Source not found: {args.source}")
    inference = build_inference(args)
    extension = os.path.splitext(args.source)[1].lower()
    if extension in {".mp4", ".avi", ".mov", ".mkv"}:
        run_video(inference, args.source, args.output)
    else:
        run_image(inference, args.source, args.output)


if __name__ == "__main__":
    main()