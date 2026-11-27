"""Model runners: onnxruntime (laptop CPU/GPU, board CPU) and the Amlogic A311D NPU.

Every runner is called the same way - ``outputs = runner(input0, input1, ...)`` with
NumPy arrays in the layout the ONNX model expects - so the OCR and detector code does
not care where a model runs. ``config/models/vehicle_text.yaml`` picks the backend per
model (its ``onnx:`` / ``nb:`` blocks).

Ported unchanged from EdgeOCR (fasttext_variant/edgeocr/runtime.py); the NPU findings
below were measured on the Electron board. ``export/...`` references point to that
project's export scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OrtRunner:
    def __init__(self, path: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        if threads:
            options.intra_op_num_threads = threads
        # By default idle onnxruntime threads busy-wait at 100% CPU. With several models
        # sharing a 6-core board that starves the other models, so let them sleep.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.name = f"onnxruntime:{providers[0].removesuffix('ExecutionProvider')}:{path.name}"

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, dict(zip(self.input_names, inputs)))


def _per_channel(value) -> np.float32 | np.ndarray:
    """Scalar -> float32; list (one value per channel) -> C x 1 x 1 array for CHW data."""
    if isinstance(value, (list, tuple)):
        return np.asarray(value, np.float32).reshape(-1, 1, 1)
    return np.float32(value)


class NpuInput:
    """How an NCHW batch-1 model input is handed to the KSNN/ASNN library.

    The library's ``NPU_SetInput_*`` reads the buffer as float32, rearranges it according
    to ``platform`` and ``reorder`` and quantizes it into the model's input tensor.
    ``scale``/``offset`` undo the ONNX model's normalization (the NPU models take raw
    0-255 pixels); ``layout`` is the order of the float buffer ("hwc" or "chw").

    Measured on the Electron board (asnn v1.4, identity-model probe): with reorder
    "2 1 0" the buffer is copied verbatim, so it must already be planar CHW like the
    ONNX input, and outputs come back NCHW. reorder "0 1 2" scrambles the data.
    ``pad`` (height, width) zero-pads the image at the bottom/right: NPU_SetInput_I16
    corrupts memory for non-square inputs, so those models take a square input and cut
    the top-left part out themselves (export/make_npu_onnx.py, PARseq encoders).
    """

    def __init__(self, scale: float = 1.0, offset: float = 0.0, layout: str = "chw",
                 platform: str | None = "ONNX", reorder: str = "2 1 0",
                 pad: tuple[int, int] | None = None, stretch: int = 1) -> None:
        if layout not in ("hwc", "chw"):
            raise ValueError(f"NPU input layout must be 'hwc' or 'chw', not {layout!r}")
        # A number, or one value per channel (list) for per-channel normalization.
        self.scale, self.offset = _per_channel(scale), _per_channel(offset)
        self.layout, self.platform, self.reorder = layout, platform, reorder
        self.pad = tuple(pad) if pad else None
        # Square input without a crop: every row repeated `stretch` times (48x336 -> 336x336);
        # the NPU model averages them back first (export/make_npu_onnx.py STRETCH).
        self.stretch = int(stretch)

    @classmethod
    def from_spec(cls, spec: dict) -> "NpuInput":
        return cls(spec.get("input_scale", 1.0), spec.get("input_offset", 0.0),
                   spec.get("input_layout", "chw"), spec.get("platform", "ONNX"),
                   spec.get("reorder", "2 1 0"), spec.get("input_pad"),
                   spec.get("input_stretch", 1))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raw = x[0]
        if np.any(self.scale != 1) or np.any(self.offset != 0):
            raw = raw * self.scale + self.offset
        if self.stretch > 1:
            raw = np.repeat(raw, self.stretch, axis=1)  # CHW: rows
        if self.pad:
            c, h, w = raw.shape
            padded = np.zeros((c, *self.pad), np.float32)
            padded[:, :h, :w] = raw
            raw = padded
        if self.layout == "hwc":
            raw = raw.transpose(1, 2, 0)
        return np.ascontiguousarray(raw, dtype=np.float32)


def _read_output(data, nchw_shape: list[int], layout: str) -> np.ndarray:
    """Flat NPU output -> the ONNX model's NCHW shape (``layout``: order of the flat data)."""
    flat = np.asarray(data, dtype=np.float32)
    if layout == "nchw":
        return flat.reshape(nchw_shape)
    n, c, *spatial = nchw_shape  # nhwc: channels last
    return np.moveaxis(flat.reshape(n, *spatial, c), -1, 1)


class KsnnRunner:
    """Amlogic A311D NPU via Khadas KSNN (``pip install ksnn-*.whl`` from Khadas).

    UNTESTED TEMPLATE - written without board access. The .nb model and its .so library
    come from converting the ONNX file with the Amlogic NPU SDK (see NPU.md).
    """

    def __init__(self, nb_path: Path, library: Path, output_shapes: list[list[int]],
                 npu_input: NpuInput | None = None, output_layout: str = "nchw") -> None:
        from ksnn.api import KSNN
        from ksnn.types import output_format

        self.net = KSNN("VIM3")
        self.net.nn_init(library=str(library), model=str(nb_path), level=0)
        self.output_shapes = output_shapes
        self.output_format = output_format.OUT_FORMAT_FLOAT32
        self.npu_input = npu_input or NpuInput()
        self.output_layout = output_layout
        self.name = f"ksnn:{nb_path.name}"

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        data = [self.npu_input(x) for x in inputs]
        outputs = self.net.nn_inference(data, platform=self.npu_input.platform,
                                        reorder=self.npu_input.reorder,
                                        output_tensor=len(self.output_shapes),
                                        output_format=self.output_format)
        return [_read_output(o, shape, self.output_layout)
                for o, shape in zip(outputs, self.output_shapes)]


class AsnnRunner:
    """Amlogic Electron NPU via the board's ``asnn`` Python API."""

    def __init__(self, nb_path: Path, library: Path, output_shapes: list[list[int]],
                 npu_input: NpuInput | None = None, output_layout: str = "nchw") -> None:
        try:
            from asnn.api import asnn
            from asnn.types import asnn_stat, output_format
        except ImportError as exc:
            raise RuntimeError(
                "The a311d profile needs the Electron ASNN package in this Python environment."
            ) from exc

        self.net = asnn()  # The Electron package selects BOARD_Electron by default.
        status = self.net.nn_init(library=str(library), model=str(nb_path), level=0)
        if status != asnn_stat.STAT_SUCCESS:
            raise RuntimeError(f"ASNN could not initialize NPU model {nb_path}")
        self.output_shapes = output_shapes
        self.output_format = output_format.OUT_FORMAT_FLOAT32
        self.npu_input = npu_input or NpuInput()
        self.output_layout = output_layout
        self.name = f"asnn:{nb_path.name}"

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        # asnn's NPU_SetInput_I8/U8/I16 read every input buffer as float32.
        data = [self.npu_input(x) for x in inputs]
        outputs = self.net.nn_inference(
            data, platform=self.npu_input.platform, reorder=self.npu_input.reorder,
            input_tensor=len(data), output_tensor=len(self.output_shapes),
            output_format=self.output_format,
        )
        return [_read_output(o, shape, self.output_layout)
                for o, shape in zip(outputs, self.output_shapes)]


class YoloDflRunner:
    """YOLO11 whose last step (DFL box decoding) runs here in NumPy instead of on the NPU.

    The NPU converter gets DFL's reshape/transpose/softmax axes wrong (class scores came
    out right, box sizes wrong), so the NPU model stops before it and returns the raw box
    distribution (1 x 64 x A) and the class scores (1 x 80 x A). This rebuilds the usual
    (1 x 84 x A) output: cx, cy, w, h in input pixels, then the 80 scores.
    """

    def __init__(self, npu, strides: tuple[int, ...] = (8, 16, 32)) -> None:
        self.npu = npu
        self.name = f"{npu.name}+dfl"
        anchors = npu.output_shapes[0][-1]
        size = round((anchors * 1024 / 21) ** 0.5)  # A = size^2 * (1/64 + 1/256 + 1/1024)
        points, steps = [], []
        for stride in strides:
            n = size // stride
            ys, xs = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
            points.append(np.stack([xs.ravel(), ys.ravel()]))
            steps.append(np.full(n * n, stride))
        self.anchors = np.concatenate(points, 1).astype(np.float32)   # 2 x A, grid units
        self.strides = np.concatenate(steps).astype(np.float32)       # A
        self.bins = np.arange(16, dtype=np.float32)[None, :, None]

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        outputs = sorted(self.npu(*inputs), key=lambda o: o.shape[1])  # 64 box bins, 80 classes
        dist, scores = outputs[0][0], outputs[1][0]
        d = dist.reshape(4, 16, -1)
        d = np.exp(d - d.max(1, keepdims=True))
        d = (d * self.bins).sum(1) / d.sum(1)                          # 4 x A: left, top, right, bottom
        x1y1, x2y2 = self.anchors - d[:2], self.anchors + d[2:]
        boxes = np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1]) * self.strides
        return [np.concatenate([boxes, scores])[None].astype(np.float32)]


class TailRunner:
    """NPU model whose last part runs on the CPU (onnxruntime), e.g. PP-OCR recognizers.

    The NPU computes the CNN (most of the work); the attention tail, which loses its
    character scores in int16 on this NPU, runs in float on the CPU. The NPU outputs are
    matched to the tail's inputs by size (they all differ).
    """

    def __init__(self, npu, tail: Path, threads: int | None) -> None:
        self.npu = npu
        self.tail = OrtRunner(tail, threads)
        self.shapes = [list(i.shape) for i in self.tail.session.get_inputs()]
        self.by_size = {int(np.prod(s)): k for k, s in enumerate(self.shapes)}
        self.name = f"{npu.name}+cpu:{tail.name}"

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        feats: list = [None] * len(self.shapes)
        for out in self.npu(*inputs):
            k = self.by_size[out.size]
            feats[k] = out.reshape(self.shapes[k])
        return self.tail(*feats)


def is_npu(runner) -> bool:
    """NPU runners (also wrapped ones) take one image per call."""
    return isinstance(runner, (AsnnRunner, KsnnRunner)) or \
        isinstance(getattr(runner, "npu", None), (AsnnRunner, KsnnRunner))


def make_runner(spec: dict, models_dir: Path, onnx_file: str, threads: int | None,
                output_shapes: list[list[int]] | None = None):
    """Build the runner a config entry asks for, e.g. {"backend": "onnxruntime", "threads": 4}.

    ``threads`` in the entry overrides the global setting; ``file`` swaps the ONNX file.
    NPU entries may set ``input_scale``/``input_offset`` (the NPU model gets
    ``input * input_scale + input_offset``, e.g. 255 for YOLO's 0-1 input) and how the
    library reads the input: ``input_layout`` ("hwc"/"chw"), ``platform``, ``reorder``,
    and the order of its outputs: ``output_layout`` ("nchw"/"nhwc").
    """
    backend = spec.get("backend", "onnxruntime")
    onnx_file = spec.get("file", onnx_file)
    threads = spec.get("threads", threads)
    if backend == "onnxruntime":
        return OrtRunner(models_dir / onnx_file, threads)
    if backend in ("ksnn", "asnn"):
        output_shapes = spec.get("output_shapes", output_shapes)  # e.g. PARseq encoders
        if output_shapes is None and spec.get("postprocess") != "cpu_tail":
            raise ValueError(f"{onnx_file}: the NPU backend needs fixed output shapes")
        runner = KsnnRunner if backend == "ksnn" else AsnnRunner
        dfl = spec.get("postprocess") == "yolo_dfl"
        tail = spec.get("postprocess") == "cpu_tail"
        if dfl:  # NPU model returns box bins and class scores; channel count from the data
            output_shapes = [[1, -1, output_shapes[0][-1]]] * 2
        if tail:  # NPU part's outputs are reshaped for the tail by TailRunner
            output_shapes = [[1, -1]] * spec.get("npu_outputs", 2)
        npu = runner(models_dir / spec["nb"], models_dir / spec["library"], output_shapes,
                     NpuInput.from_spec(spec), spec.get("output_layout", "nchw"))
        if tail:
            return TailRunner(npu, models_dir / spec["tail"], spec.get("tail_threads", threads))
        return YoloDflRunner(npu) if dfl else npu
    raise ValueError(f"Unknown backend {backend!r} for {onnx_file}")
