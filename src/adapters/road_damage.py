"""
Road Damage Detector Adapter
Adapter for RoadScanCV road damage object detection (potholes, cracks, manholes).

IMPORTANT: Requires custom trained weights (RoadScanCV).
Do NOT download or substitute generic COCO weights for road damage detection.
"""
import os
from typing import Dict, Any, List
import numpy as np

from .base import BaseModelAdapter

ROAD_DAMAGE_CLASSES = {
    0: "pothole",
    1: "crack",
    2: "manhole"
}

class RoadDamageDetectorAdapter(BaseModelAdapter):
    """
    Adapter for RoadScanCV road damage model (pothole, crack, manhole).
    """

    def __init__(self, weights_path: str = "weights/road_damage_best.pt"):
        # Check alternative default path from RoadScanCV if custom folder doesn't have it
        if not os.path.exists(weights_path) and os.path.exists("RoadScanCV/runs/A/weights/best.pt"):
            weights_path = "RoadScanCV/runs/A/weights/best.pt"
        super().__init__(weights_path=weights_path)
        self.model_name = "RoadScanCV YOLO (Pothole/Crack/Manhole)"
        self.load()

    def load(self) -> bool:
        if not self.weights_path or not os.path.exists(self.weights_path):
            self.status = "FAIL"
            self.status_message = f"Missing weights: Pretrained checkpoint not found at '{self.weights_path}'"
            self.is_loaded = False
            return False

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            self.is_loaded = True
            self.status = "PASS"
            self.status_message = f"Component loaded successfully ({self.weights_path})"
            return True
        except ImportError:
            self.status = "FAIL"
            self.status_message = "Dependency error: 'ultralytics' module not installed"
            self.is_loaded = False
            return False
        except Exception as e:
            self.status = "FAIL"
            self.status_message = f"Failed to load model checkpoint: {str(e)}"
            self.is_loaded = False
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Cannot run inference: {self.status_message}")

        results = self.model(image, verbose=False)[0]
        damage_detections: List[Dict[str, Any]] = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            label = ROAD_DAMAGE_CLASSES.get(cls_id, f"unknown_damage_{cls_id}")
            damage_detections.append({
                "class_id": cls_id,
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

        return {
            "component": "road_damage_detector",
            "model": self.model_name,
            "damage_count": len(damage_detections),
            "detections": damage_detections
        }
