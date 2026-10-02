"""
Generic Object Detector Adapter
Wraps Ultralytics YOLOv8 trained on COCO for detecting generic road objects:
cars, motorcycles, bicycles, buses, trucks, pedestrians, and traffic lights.

NOTE: This detector uses standard COCO pre-trained weights and is NOT an Indian-specific detector.
COCO weights are exclusively restricted to generic object classes.
"""
import os
from typing import Dict, Any, List
import numpy as np

from .base import BaseModelAdapter

# COCO Class IDs for generic road objects
COCO_TARGET_CLASSES = {
    0: "pedestrian",      # person
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}

class GenericObjectDetectorAdapter(BaseModelAdapter):
    """
    Adapter for generic COCO object detection (vehicles, pedestrians, traffic lights).
    """

    def __init__(self, weights_path: str = "yolov8n.pt"):
        super().__init__(weights_path=weights_path)
        self.model_name = "YOLOv8 (COCO Pre-trained)"
        self.load()

    def load(self) -> bool:
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
            self.status_message = f"Failed to load model: {str(e)}"
            self.is_loaded = False
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Cannot run inference: {self.status_message}")

        results = self.model(image, verbose=False)[0]
        detections: List[Dict[str, Any]] = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id in COCO_TARGET_CLASSES:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append({
                    "class_id": cls_id,
                    "label": COCO_TARGET_CLASSES[cls_id],
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })

        return {
            "component": "generic_object_detector",
            "model": self.model_name,
            "target_objects_count": len(detections),
            "detections": detections
        }
