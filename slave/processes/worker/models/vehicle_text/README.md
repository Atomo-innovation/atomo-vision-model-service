# Vehicle Text model files

Reads text written on vehicles (number plate, names, slogans, company text) in
English, Hindi and Gujarati. Ported from EdgeOCR (`~/EdgeOCR/fasttext_variant`,
PP-OCR engine): YOLO11n vehicles -> ByteTrack -> PP-OCR text lines -> one CTC
reader per script -> plate/text voting across frames.

Unlike the other models (weights flat in the model folder), each job type's
weights live **in its own extension folder**, next to its worker/detector code.
Files both job types use stay at the top:

```
processes/worker/models/vehicle_text/
├── vehicle_text_<script>_dict.txt   ← character list per recognizer (both job types)
├── vehicle_text_lid.ftz             ← fastText language ID (optional, off by default)
│                                      <script> = english | hindi | gujarati
│
├── onnx/                            ← .onnx job: everything on CPU (onnxruntime)
│   ├── vehicle_text.onnx              YOLO11n (COCO) vehicle detector
│   ├── vehicle_text_det.onnx          PP-OCR text-line detector (DB)
│   ├── vehicle_text_<script>.onnx     CTC recognizer
│   └── vehicle_text_onnx_worker.py, vehicle_text_onnx_detector.py
│
├── nb/                              ← .nb job: A311D NPU (asnn), int16
│   ├── vehicle_text.nb / .so          YOLO head (DFL box decoding runs on CPU)
│   ├── vehicle_text_det.nb / .so      text-line detector
│   ├── vehicle_text_<script>.nb / .so recognizer backbone ...
│   ├── vehicle_text_<script>_tail.onnx  ... + the CPU tail that finishes it
│   └── vehicle_text_nb_worker.py, vehicle_text_nb_detector.py
│
├── ocr_runtime.py       onnxruntime / ASNN runners (board-validated, keep as-is)
├── vehicle_detector.py  YOLO pre/post-processing
├── vehicle_tracker.py   ByteTrack
├── ppocr_reader.py      text-line detection + per-script reading
├── ocr_scheduler.py     when to OCR each vehicle; background OCR thread
├── plate_parser.py      Indian plate parsing + cross-frame plate voting
├── ocr_types.py         Word / Line / OCRResult
├── text_overlay.py      draws boxes + Hindi/Gujarati text (needs Noto fonts)
└── language_id.py       fastText wrapper
```

The job's `model_name` picks the job type, and is resolved inside that type's
folder:

```yaml
model_name: vehicle_text.onnx   # -> onnx/vehicle_text.onnx, everything on CPU
model_name: vehicle_text.nb     # -> nb/vehicle_text.nb, YOLO + text detector +
                                #    recognizer backbones on the NPU
```

All other files (text detector, recognizers, tails, NPU libraries) are named in
`config/models/vehicle_text.yaml` and resolved against the same folder.

There is no `.pt` or standalone `.so` job: EdgeOCR ships no PyTorch weights, and
its `.so` files are only the `.nb` companion libraries.

Language ID needs `fasttext-wheel`, which requires `numpy<2`; it is disabled by
default. Each text line still carries the script it was read with.
