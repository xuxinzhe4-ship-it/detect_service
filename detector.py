# detector.py
# ONNXRuntime inference utilities for a YOLO-style detector (letterbox + NMS in ONNX).
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import base64
import cv2
import numpy as np
import onnxruntime as ort


class ONNXDetector1600NMS:
    """
    ONNX exported by: yolo export ... format=onnx opset=21 nms=True (fixed input 1600x1600)
    input : images [1,3,1600,1600] float32, RGB, 0-1
    output: output0 [1,300,6] float32 -> [x1,y1,x2,y2,score,cls] in letterbox coords
    """

    def __init__(
        self,
        onnx_path: str,
        class_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        imgsz: int = 1600,
    ):
        self.imgsz = int(imgsz)
        self.class_names = class_names or ["defect"]

        avail = ort.get_available_providers()
        if providers is None:
            prefer = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in prefer if p in avail] or avail
        else:
            providers = [p for p in providers if p in avail] or avail

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _letterbox(img_bgr: np.ndarray, new: int, color=(114, 114, 114)) -> Tuple[np.ndarray, float, float, float]:
        """Resize + pad to (new,new). Returns (img, r, dw, dh)."""
        h0, w0 = img_bgr.shape[:2]
        r = min(new / h0, new / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))

        img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw = (new - new_w) / 2.0
        dh = (new - new_h) / 2.0

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, dw, dh

    @staticmethod
    def _scale_back_xyxy(
        boxes: np.ndarray, r: float, dw: float, dh: float, orig_w: int, orig_h: int
    ) -> np.ndarray:
        """Map xyxy from letterbox coords back to original image coords."""
        b = boxes.copy()
        b[:, [0, 2]] -= dw
        b[:, [1, 3]] -= dh
        b[:, :4] /= r
        b[:, 0] = np.clip(b[:, 0], 0, orig_w)
        b[:, 2] = np.clip(b[:, 2], 0, orig_w)
        b[:, 1] = np.clip(b[:, 1], 0, orig_h)
        b[:, 3] = np.clip(b[:, 3], 0, orig_h)
        return b

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image")

        orig_h, orig_w = img_bgr.shape[:2]
        lb, r, dw, dh = self._letterbox(img_bgr, self.imgsz)

        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW

        meta = {"orig_w": orig_w, "orig_h": orig_h, "r": r, "dw": dw, "dh": dh}
        return x, meta

    def postprocess(self, out0: np.ndarray, meta: Dict[str, Any], conf: float) -> List[Dict[str, Any]]:
        det = np.asarray(out0)
        if det.ndim == 3:
            det = det[0]  # [300,6]

        # rows with score==0 are padding; filter by conf
        scores = det[:, 4]
        keep = scores > float(conf)
        det = det[keep]
        if det.shape[0] == 0:
            return []

        boxes = det[:, 0:4]
        scores = det[:, 4]
        clss = det[:, 5].astype(int)

        boxes = self._scale_back_xyxy(boxes, meta["r"], meta["dw"], meta["dh"], meta["orig_w"], meta["orig_h"])

        results: List[Dict[str, Any]] = []
        for b, s, c in zip(boxes, scores, clss):
            name = self.class_names[c] if 0 <= c < len(self.class_names) else str(c)
            results.append(
                {
                    "cls_id": int(c),
                    "cls_name": name,
                    "conf": float(s),
                    "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                }
            )
        results.sort(key=lambda x: x["conf"], reverse=True)
        return results

    def predict(self, img_bgr: np.ndarray, conf: float = 0.293) -> List[Dict[str, Any]]:
        x, meta = self.preprocess(img_bgr)
        out0 = self.session.run(None, {self.input_name: x})[0]  # output0
        return self.postprocess(out0, meta, conf=conf)


def b64_to_bgr(image_b64: str) -> np.ndarray:
    """Supports 'data:image/...;base64,...' or pure base64."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


def bgr_to_b64_jpg(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def draw_dets(img_bgr: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f'{d["cls_name"]}:{d["conf"]:.2f}',
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return out
