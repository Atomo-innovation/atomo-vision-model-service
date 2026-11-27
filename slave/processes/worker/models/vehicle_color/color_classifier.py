"""Universal Vehicle Color Classifier.

Standalone classification engine for vehicle color recognition.
Designed to run independently (CLI, scripts, external apps) with zero
dependencies on NATS, MinIO, or Slave node frameworks, as well as integrated
inside atomo-vision-model-service via VehicleColorEnricher.

Supported model formats:
- .pt   : PyTorch via Ultralytics YOLO classification
- .onnx : ONNX Runtime
- .nb   : Amlogic A311D NPU via ASNN/KSNN runtime
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from slave.runtimes.common import runtime_utils as ru

# 9 target vehicle color classes matching the trained model
DEFAULT_COLOR_NAMES: Dict[int, str] = {
    0: "black",
    1: "blue",
    2: "brown",
    3: "gray",
    4: "green",
    5: "orange",
    6: "red",
    7: "white",
    8: "yellow",
}


class VehicleColorClassifier:
    """Universal Vehicle Color Classifier supporting .pt, .onnx, and .nb models."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        class_names: Optional[Dict[int, str]] = None,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Vehicle color model not found: {model_path}")

        self._model_path = os.path.abspath(model_path)
        self._device = None if device == "auto" else device
        self._input_size = input_size
        self._class_names = class_names or DEFAULT_COLOR_NAMES

        _, ext = os.path.splitext(self._model_path)
        self._ext = ext.lower()

        self._model = None
        self._backend = None
        self._init_runtime()

    def _init_runtime(self) -> None:
        if self._ext == ".pt":
            self._backend = "ultralytics"
            from ultralytics import YOLO  # lazy import

            self._model = YOLO(self._model_path)
            if hasattr(self._model, "names") and isinstance(self._model.names, dict):
                self._class_names = self._model.names

        elif self._ext == ".onnx":
            self._backend = "onnx"
            import onnxruntime as ort  # lazy import

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device != "cpu"
                else ["CPUExecutionProvider"]
            )
            self._model = ort.InferenceSession(self._model_path, providers=providers)
            self._input_name = self._model.get_inputs()[0].name

        elif self._ext in (".nb", ".so"):
            self._backend = "npu"
            dir_name = os.path.dirname(self._model_path)
            lib_candidates = [
                os.path.join(dir_name, "libnn_vehicle_color.so"),
                os.path.join(dir_name, "vehicle_color.so"),
                os.path.splitext(self._model_path)[0] + ".so",
            ]
            library_path = next((p for p in lib_candidates if os.path.isfile(p)), None)
            try:
                from slave.runtimes.npu.nb_runtime import NbRuntime
                self._model = NbRuntime(
                    self._model_path, library_path=library_path, board="Electron", output_tensor=1,
                )
            except Exception:
                from asnn.api import asnn
                self._model = asnn("Electron")
                self._model.nn_init(library=library_path, model=self._model_path, level=0)
        else:
            raise ValueError(
                f"Unsupported model extension '{self._ext}'. Expected .pt, .onnx, or .nb/.so"
            )

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify the color of a single cropped vehicle image (BGR format).

        Returns:
            Tuple of (color_name, confidence_score) e.g. ("white", 0.94)
        """
        if crop is None or crop.size == 0:
            return "unknown", 0.0

        if self._backend == "ultralytics":
            kwargs = {
                "source": crop,
                "device": self._device,
                "verbose": False,
            }
            results = self._model.predict(**kwargs)
            probs = results[0].probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf.item())
            color_name = self._class_names.get(top1_idx, str(top1_idx))
            return color_name, top1_conf

        elif self._backend == "onnx":
            resized = cv2.resize(crop, self._input_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (rgb - mean) / std
            tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

            outputs = self._model.run(None, {self._input_name: tensor})
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            top1_idx = int(np.argmax(probs))
            top1_conf = float(probs[top1_idx])
            color_name = self._class_names.get(top1_idx, str(top1_idx))
            return color_name, top1_conf

        elif self._backend == "npu":
            resized = cv2.resize(crop, self._input_size)
            # The SDK reads the input as float32, so pass the same RGB CHW
            # [0, 1] tensor the .nb detectors use; raw uint8 HWC is 4x too
            # small and makes the SDK read past the buffer.
            outputs = self._model.infer(ru.to_input_tensor(resized))
            logits = outputs[0].flatten()
            if np.all(logits >= 0) and abs(float(np.sum(logits)) - 1.0) < 1e-2:
                # The .nb export already ends in softmax; applying it again
                # flattens confidences to <=0.25 and fails confidence_threshold.
                probs = logits
            else:
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
            top1_idx = int(np.argmax(probs))
            top1_conf = float(probs[top1_idx])
            color_name = self._class_names.get(top1_idx, str(top1_idx))
            return color_name, top1_conf

        return "unknown", 0.0

    def classify_batch(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Classify a list of cropped vehicle images."""
        if not crops:
            return []
        if self._backend == "ultralytics":
            valid_crops = [c for c in crops if c is not None and c.size > 0]
            if not valid_crops:
                return [("unknown", 0.0)] * len(crops)
            results = self._model.predict(
                source=valid_crops,
                device=self._device,
                verbose=False,
            )
            out = []
            res_idx = 0
            for c in crops:
                if c is None or c.size == 0:
                    out.append(("unknown", 0.0))
                else:
                    probs = results[res_idx].probs
                    top1_idx = int(probs.top1)
                    top1_conf = float(probs.top1conf.item())
                    color_name = self._class_names.get(top1_idx, str(top1_idx))
                    out.append((color_name, top1_conf))
                    res_idx += 1
            return out

        return [self.classify(crop) for crop in crops]

    def release(self) -> None:
        """Release underlying model resources."""
        self._model = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Vehicle Color Classifier")
    parser.add_argument("--image", required=True, help="Path to cropped vehicle image")
    parser.add_argument(
        "--weights",
        default=os.path.join(os.path.dirname(__file__), "vehicle_color.pt"),
        help="Path to vehicle color model weights (.pt, .onnx, .nb)",
    )
    parser.add_argument("--device", default="auto", help="Compute device: auto, cpu, cuda:0")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to read image '{args.image}'.")
        sys.exit(1)

    classifier = VehicleColorClassifier(args.weights, device=args.device)
    color, conf = classifier.classify(image)
    print(f"Prediction: {color} (Confidence: {conf:.4f})")


if __name__ == "__main__":
    main()
