[English](README.md) | [中文](README.zh-CN.md)

# detect_service — Power-grid Object Detection Inference Service (ONNX)

Last updated: 2026-03-03

This repository is an internship practice project created by a beginner to demonstrate engineering progress. Some files are not provided due to the use of an internal platform and dataset.  
This service is a lightweight HTTP inference server built with Flask and ONNX Runtime. It performs object detection on power-grid related images, focusing on detecting and localizing the defect “missing insulation cover on the tension clamp” (耐张线夹绝缘罩缺失), and returns a list of bounding boxes.

## 1. Task Goal

- Input: one power-grid related image
- Output: a list of detections for the defect missing insulation cover on the tension clamp (box coordinates, class, confidence)

Default class:

- `defect` (single-class example, representing the missing-cover defect)

## 2. API Overview

| Method | Path | Request Type | Usage | Response Type |
|---|---|---|---|---|
| GET | `/health` | None | Health check | JSON |
| POST | `/detect` | `application/json` (single image in Base64) | Single-image detection | JSON |

---

## 3. Model Weights

`best.onnx` is NOT included in this repository because it was trained with an internal dataset.

- Default path: `./best.onnx` next to `app.py`
- Provide your own weights as `best.onnx`, or change `onnx_path` in code to point to your model

## 4. Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place best.onnx in the current directory, then start the server
python app.py
```

Base URL placeholder: `http://<HOST>:<PORT>`  
Example: `http://127.0.0.1:8000`

## 5. GET /health

**Request:**
```bash
curl http://<HOST>:<PORT>/health
```

**Response (JSON):**
```json
{"status":"ok"}
```

## 6. POST /detect (Object Detection)

### 6.1 Request
- Content-Type: `application/json`
- JSON body fields:
  - `image_b64` (required): image Base64 string (supports `data:image/...;base64,` prefix or pure Base64)
  - `conf` (optional): confidence threshold, default `0.293`
  - `return_annotated` (optional): whether to return an annotated JPG in Base64, default `false`

### 6.2 Response (JSON)
```json
{
  "count": 1,
  "detections": [
    {
      "cls_id": 0,
      "cls_name": "defect",
      "conf": 0.91,
      "xyxy": [123.0, 45.0, 456.0, 300.0]
    }
  ],
  "conf": 0.293,
  "time_ms": 18.5,
  "image_w": 1920,
  "image_h": 1080,
  "annotated_b64_jpg": "<optional>"
}
```

Notes:
- `detections[].xyxy` uses pixel coordinates in `x1,y1,x2,y2` format
- `annotated_b64_jpg` is only returned when `return_annotated=true`

### 6.3 Example (Linux)

```bash
curl -X POST http://<HOST>:<PORT>/detect   -H 'Content-Type: application/json'   -d '{"image_b64":"'"$(base64 -w0 /path/to/test.jpg)"'","conf":0.3,"return_annotated":false}'
```

---

## 7. Docker

`Dockerfile` is a public template and requires a base image you can access.

Build:

```bash
docker build --build-arg BASE_IMAGE=python:3.9-slim -t detect_service:latest .
```

Run (recommended to mount weights at runtime):

```bash
docker run --rm -p 8000:8000   -v "$PWD/best.onnx:/app/best.onnx:ro"   detect_service:latest
```
