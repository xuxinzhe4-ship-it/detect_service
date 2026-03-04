[English](README.md) | [中文](README.zh-CN.md)

# detect_service — 电网目标检测推理服务（ONNX）

最后更新：2026.3.3

该仓库为本人作为初学者实习期间的模拟训练，用于展示工程化学习成果。由于涉及内部平台与数据集训练，本仓库未提供部分文件。本服务为一个轻量的 HTTP 推理服务，基于 Flask 与 ONNX Runtime。用于对电网相关图片进行目标检测，重点发现并定位“**耐张线夹绝缘罩缺失**”缺陷，返回缺陷检测框列表。

## 1. 任务目标

- 输入：一张电网相关图片
- 输出：**耐张线夹绝缘罩缺失**缺陷的检测框列表（坐标、类别、置信度）

默认类别：

- `defect`（单类示例，对应“耐张线夹绝缘罩缺失”）

## 2. 接口总览

| Method | Path | 请求类型 | 主要用途 | 响应类型 |
|---|---|---|---|---|
| GET | `/health` | 无 | 健康检查 | JSON |
| POST | `/detect` | `application/json`（Base64 单张图片） | 单张图片目标检测 | JSON |

---

## 3. 权重文件

由于使用内部数据集训练，本仓库不提供 `best.onnx`。

- 默认路径：`./best.onnx`（与 `app.py` 同目录）
- 请自行准备并放置为 `best.onnx`，或修改代码中的 `onnx_path` 指向你的权重路径

## 4. 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 将 best.onnx 放在当前目录后启动
python app.py
```

Base URL 占位符：`http://<HOST>:<PORT>`  
示例：`http://127.0.0.1:8000`

## 5. GET /health

**请求：**
```bash
curl http://<HOST>:<PORT>/health
```

**响应（JSON）：**
```json
{"status":"ok"}
```

## 6. POST /detect（目标检测）

### 6.1 请求
- Content-Type：`application/json`
- Body（JSON）字段：
  - `image_b64`（必填）：图片 Base64 字符串（支持 `data:image/...;base64,` 前缀或纯 Base64）
  - `conf`（可选）：置信度阈值，默认 `0.293`
  - `return_annotated`（可选）：是否返回带框可视化的 JPG Base64，默认 `false`

### 6.2 响应（JSON）
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

字段说明：
- `detections[].xyxy`：检测框坐标（左上 x1,y1 右下 x2,y2），单位为像素
- `annotated_b64_jpg`：仅当 `return_annotated=true` 时返回

### 6.3 示例（Linux）

```bash
curl -X POST http://<HOST>:<PORT>/detect   -H 'Content-Type: application/json'   -d '{"image_b64":"'"$(base64 -w0 /path/to/test.jpg)"'","conf":0.3,"return_annotated":false}'
```

---

## 7. Docker

`Dockerfile` 为公开模板，需要你传入一个自己可访问的基础镜像。

构建：

```bash
docker build --build-arg BASE_IMAGE=python:3.9-slim -t detect_service:latest .
```

运行（推荐通过挂载方式提供权重）：

```bash
docker run --rm -p 8000:8000   -v "$PWD/best.onnx:/app/best.onnx:ro"   detect_service:latest
```
