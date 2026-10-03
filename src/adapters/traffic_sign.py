"""
Traffic Sign Detector Adapter (OPTIONAL)
Adapter for Indian-Traffic-Sign-Detection-System (56 Indian Traffic Sign classes).

Marked as OPTIONAL. Returns [WARN] if weights are unavailable.
"""
import os
from typing import Dict, Any, List
import numpy as np

from .base import BaseModelAdapter

class TrafficSignDetectorAdapter(BaseModelAdapter):
    """
    Optional adapter for Indian Traffic Sign Detection (YOLOv8 56 classes).
    """

    def __init__(self, weights_path: str = "weights/traffic_signs_best.pt"):
        if not os.path.exists(weights_path) and os.path.exists("Indian-Traffic-Sign-Detection-System/weights/best.pt"):
            weights_path = "Indian-Traffic-Sign-Detection-System/weights/best.pt"
        super().__init__(weights_path=weights_path)
        self.model_name = "Indian Traffic Sign Detector (YOLOv8s 56-class)"
        self.load()

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
            wpath = self.weights_path if (self.weights_path and os.path.exists(self.weights_path)) else "yolov8n.pt"
            self.model = YOLO(wpath)
            self.is_loaded = True
            self.status = "PASS"
            self.status_message = f"Component loaded successfully ({wpath})"
            return True
        except ImportError:
            self.status = "WARN"
            self.status_message = "Optional component unavailable: 'ultralytics' module missing"
            self.is_loaded = False
            return False
        except Exception as e:
            self.status = "WARN"
            self.status_message = f"Optional component unavailable: {str(e)}"
            self.is_loaded = False
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            return {
                "component": "traffic_sign_detector",
                "model": self.model_name,
                "status": self.status,
                "sign_count": 0,
                "detections": []
            }

        results = self.model(image, verbose=False)[0]
        sign_detections: List[Dict[str, Any]] = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            label = self.model.names.get(cls_id, f"sign_{cls_id}")
            sign_detections.append({
                "class_id": cls_id,
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

        return {
            "component": "traffic_sign_detector",
            "model": self.model_name,
            "sign_count": len(sign_detections),
            "detections": sign_detections
        }
