#!/usr/bin/env python3
"""Standalone Vehicle Color Recognition & Benchmarking Tool.

Supports both:
1. End-to-end full scene images (runs vehicle detector -> crops vehicles -> classifies colors)
2. Direct vehicle crops (classifies color directly on pre-cropped vehicle images)

Usage examples:
    # Full scene image (auto-detects vehicles, crops, and classifies their colors):
    python tools/vehicle_color.py --image path/to/scene.jpg --detect

    # Direct crop of a car:
    python tools/vehicle_color.py --image path/to/car_crop.jpg

    # Directory of vehicle crops:
    python tools/vehicle_color.py --dir path/to/vehicle_crops/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Allow running from root repo or tools/ folder
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slave.processes.worker.models.vehicle_color.color_classifier import (
    DEFAULT_COLOR_NAMES,
    VehicleColorClassifier,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_single_image(
    classifier: VehicleColorClassifier,
    image_path: str,
    detect: bool = False,
    detector_weights: Optional[str] = None,
    conf_thresh: float = 0.0,
    output_path: Optional[str] = None,
) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'")
        return

    h, w = img.shape[:2]

    if detect:
        if not detector_weights or not os.path.exists(detector_weights):
            print(f"Error: Detector model weights not found at: {detector_weights}")
            return

        from ultralytics import YOLO

        print(f"Detecting vehicles using: {detector_weights}...")
        detector = YOLO(detector_weights)
        res = detector.predict(img, verbose=False)[0]

        boxes = res.boxes
        if len(boxes) == 0:
            print("No vehicles detected in this image.")
            return

        print(f"\nFound {len(boxes)} vehicle(s) in {image_path}:")
        annotated = img.copy()

        for idx, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            cls_name = res.names.get(cls_id, f"class_{cls_id}")
            det_conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(round(float(v))) for v in box.xyxy[0]]

            # Clamp coordinates
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)
            crop = img[y1_c:y2_c, x1_c:x2_c]

            t0 = time.perf_counter()
            color, color_conf = classifier.classify(crop)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            status = "ACCEPTED" if color_conf >= conf_thresh else "LOW CONFIDENCE"
            print(
                f"  Vehicle #{idx}: {cls_name.upper():<10s} "
                f"bbox=[{x1}, {y1}, {x2}, {y2}] "
                f"-> Color: {color:<8s} (conf: {color_conf:.4f}, {status}) "
                f"[{latency_ms:.1f} ms]"
            )

            # Annotate box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name}: {color} ({color_conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lbl_y = max(y1, th + 10)
            cv2.rectangle(
                annotated, (x1, lbl_y - th - 8), (x1 + tw + 6, lbl_y), (0, 255, 0), -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, lbl_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"\nSaved annotated image to: {output_path}")

    else:
        t0 = time.perf_counter()
        color, conf = classifier.classify(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        status = "ACCEPTED" if conf >= conf_thresh else "LOW CONFIDENCE"
        print(f"\n[Result on input crop] Image: {image_path}")
        print(f"  Dimensions: {w}x{h}")
        print(f"  Color:      {color}")
        print(f"  Confidence: {conf:.4f} ({status})")
        print(f"  Latency:    {latency_ms:.2f} ms")
        if w > 800 or h > 800:
            print(
                "\n  NOTE: This image is large and appears to be a full uncropped photo."
            )
            print(
                "  To detect and crop vehicles automatically, add the '--detect' flag:"
            )
            print(f"    python tools/vehicle_color.py --image {image_path} --detect")


def benchmark_directory(
    classifier: VehicleColorClassifier,
    dir_path: str,
    conf_thresh: float = 0.0,
    output_dir: str | None = None,
) -> None:
    p = Path(dir_path)
    if not p.is_dir():
        print(f"Error: Directory '{dir_path}' does not exist.")
        return

    files = [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    if not files:
        print(f"No image files found in '{dir_path}'")
        return

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    latencies: List[float] = []
    results = []

    print(f"\nRunning benchmark on {len(files)} vehicle crop images...")
    for idx, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            continue

        t0 = time.perf_counter()
        color, conf = classifier.classify(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        results.append((f.name, color, conf))

        if output_dir:
            annotated = img.copy()
            label = f"{color} ({conf:.2f})"
            cv2.putText(
                annotated,
                label,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imwrite(os.path.join(output_dir, f.name), annotated)

    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total Images:     {len(latencies)}")
    print(f"Mean Latency:     {avg_ms:.2f} ms")
    print(f"Min Latency:      {min(latencies):.2f} ms")
    print(f"Max Latency:      {max(latencies):.2f} ms")
    print(f"Throughput:       {throughput:.2f} images/sec")
    print("=" * 50)

    print("\nFirst 10 sample predictions:")
    for name, color, conf in results[:10]:
        print(f"  {name:30s} -> {color:10s} (conf: {conf:.4f})")


def main() -> None:
    default_color_weights = str(
        REPO_ROOT
        / "slave"
        / "processes"
        / "worker"
        / "models"
        / "vehicle_color"
        / "vehicle_color.pt"
    )
    default_vehicle_weights = str(
        REPO_ROOT
        / "slave"
        / "processes"
        / "worker"
        / "models"
        / "vehicle"
        / "vehicle.pt"
    )

    parser = argparse.ArgumentParser(
        description="Standalone Vehicle Color Recognition & Benchmark"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a vehicle image or crop")
    group.add_argument("--dir", help="Directory of vehicle crop images to benchmark")

    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run vehicle detector first to locate and crop vehicles before classifying color",
    )
    parser.add_argument(
        "--detector-weights",
        default=default_vehicle_weights,
        help="Path to vehicle detector model (used when --detect is enabled)",
    )
    parser.add_argument(
        "--weights",
        default=default_color_weights,
        help="Path to vehicle_color model (.pt, .onnx, .nb)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on ('auto', 'cpu', 'cuda:0')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Minimum confidence threshold for positive detection",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save annotated result image (when using --detect)",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory to save annotated result images (when using --dir)",
    )

    args = parser.parse_args()

    print(f"Loading vehicle color model: {args.weights}")
    classifier = VehicleColorClassifier(args.weights, device=args.device)
    print("Model loaded successfully.")

    if args.image:
        process_single_image(
            classifier,
            args.image,
            detect=args.detect,
            detector_weights=args.detector_weights,
            conf_thresh=args.conf,
            output_path=args.output,
        )
    elif args.dir:
        benchmark_directory(
            classifier,
            args.dir,
            conf_thresh=args.conf,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
