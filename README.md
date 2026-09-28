# Spatial AI prototype

A small first step toward a camera-based memory system: ingest a still image, recognize common objects, record where each object appears in that image, and keep the observation for later retrieval.

## Current architecture

```text
local image
    ↓
Ultralytics YOLO detector (default: yolo11n.pt)
    ↓
observation + object labels, confidence, and 2D image boxes
    ↓
append-only JSON Lines file (data/observations.jsonl)
```

The prototype uses a pretrained detector, so it recognizes the classes supported by the selected model. Each observation includes a timestamp, source filename and SHA-256, image dimensions, detector name, and a list of object-location records. A location stores the bounding box in pixels and normalized image coordinates (0–1), plus its normalized center. Empty detections are recorded too, so each successfully processed image leaves an observation.

These locations are **relative to the image only**. The prototype does not know camera pose, depth, or a room/world coordinate frame, and it does not yet answer questions about where an object was last seen.

## Run it

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python spatial_memory.py path/to/image.jpg
```

On the first run, Ultralytics downloads the default `yolo11n.pt` weights. Choose another compatible model or an existing weights file with `--model`; adjust the detection threshold with `--confidence`.

```bash
python spatial_memory.py kitchen.jpg \
  --output data/observations.jsonl \
  --model yolo11n.pt \
  --confidence 0.4
```

Each run appends one JSON object on its own line. The output file is local and created automatically. For example, a detected mug is represented approximately as:

```json
{
  "label": "cup",
  "confidence": 0.91,
  "location": {
    "coordinate_frame": "image",
    "bbox_xyxy_px": [120.0, 80.0, 260.0, 240.0],
    "bbox_xyxy_normalized": [0.125, 0.083333, 0.270833, 0.25],
    "center_normalized": [0.197917, 0.166667]
  }
}
```

## Next build step

Build a small retrieval layer over the JSONL log: accept an object label, search observations newest-first, and return the most recent matching image, timestamp, and location. This makes the saved records useful as a basic “where did I last see it?” memory before adding live phone-camera capture.

After retrieval, the main spatial upgrade is to attach camera pose and depth (or another distance estimate) to each observation, then map detections into a stable room coordinate frame. That is required for meaningful real-world or AR locations; normalized image boxes alone cannot provide them.

## Project shape

- `spatial_memory.py` — image validation, detector call, record creation, and JSONL persistence.
- `requirements.txt` — minimal runtime dependencies.
- `data/observations.jsonl` — generated local observation log (not committed).

The detector's output is a prediction, not a guaranteed identification. Keep the source image hash and confidence with each record so later retrieval can expose provenance and uncertainty.
