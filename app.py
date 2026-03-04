# app.py
# Lightweight Flask server for ONNXRuntime object detection.
import time
from flask import Flask, request, jsonify

from detector import ONNXDetector1600NMS, b64_to_bgr, bgr_to_b64_jpg, draw_dets

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB request size limit

# Load the ONNX detector once at startup.
MODEL = ONNXDetector1600NMS(
    onnx_path="best.onnx",
    class_names=["defect"],
    imgsz=1600,
)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/detect")
def detect():
    t0 = time.time()
    payload = request.get_json(silent=True) or {}

    image_b64 = payload.get("image_b64")
    if not image_b64:
        return jsonify({"error": "missing field: image_b64"}), 400

    conf = float(payload.get("conf", 0.293))
    return_annotated = bool(payload.get("return_annotated", False))

    try:
        img = b64_to_bgr(image_b64)
        dets = MODEL.predict(img, conf=conf)

        resp = {
            "count": len(dets),
            "detections": dets,
            "conf": conf,
            "time_ms": round((time.time() - t0) * 1000, 2),
        }

        resp["image_w"] = int(img.shape[1])
        resp["image_h"] = int(img.shape[0])

        if return_annotated:
            vis = draw_dets(img, dets)
            resp["annotated_b64_jpg"] = bgr_to_b64_jpg(vis)

        return jsonify(resp)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
