"""
Image Pipeline Module
Executes end-to-end multi-model inference, contextual spatial analysis,
severity calculation, hazard scoring, annotation, and JSON report generation on a single image.
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple

from src.adapters import (
    GenericObjectDetectorAdapter,
    RoadDamageDetectorAdapter,
    RoadSegmenterAdapter,
    TrafficSignDetectorAdapter
)
from src.analysis import PotholeSeverityAnalyzer
from src.analysis.spatial_analysis import SpatialAnalyzer
from src.analysis.hazard_indicator import RoadHazardIndicatorCalculator
from src.pipeline.visualizer import Visualizer
from src.pipeline.json_reporter import JSONReporter

class ImagePipeline:
    """
    Unified multi-model single image analysis pipeline.
    """

    def __init__(self):
        # Instantiate model adapters once
        self.object_detector = GenericObjectDetectorAdapter()
        self.road_damage_detector = RoadDamageDetectorAdapter()
        self.road_segmenter = RoadSegmenterAdapter()
        self.traffic_sign_detector = TrafficSignDetectorAdapter()

        # Instantiate analysis engines
        self.pothole_severity_analyzer = PotholeSeverityAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.hazard_calculator = RoadHazardIndicatorCalculator()
        self.visualizer = Visualizer()
        self.json_reporter = JSONReporter()

    def get_component_statuses(self) -> Dict[str, str]:
        """Returns component status dictionary."""
        return {
            "generic_object_detector": self.object_detector.get_status()[0],
            "road_damage_detector": self.road_damage_detector.get_status()[0],
            "road_segmenter": self.road_segmenter.get_status()[0],
            "traffic_sign_detector": self.traffic_sign_detector.get_status()[0],
        }

    def process(self, image: np.ndarray, source_name: str = "uploaded_image.jpg") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes image frame and returns (annotated_bgr_image, json_report_dict).
        """
        h, w = image.shape[:2]

        # --- Performance: Resize for inference, render on original ---
        INFER_MAX = 640
        scale = min(INFER_MAX / w, INFER_MAX / h, 1.0)  # Never upscale
        if scale < 1.0:
            infer_w, infer_h = int(w * scale), int(h * scale)
            infer_frame = cv2.resize(image, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_frame = image
            scale = 1.0

        # 1. Generic Object Detection (COCO)
        objects = []
        if self.object_detector.is_loaded:
            try:
                res = self.object_detector.predict(infer_frame)
                objects = self._scale_detections(res.get("detections", []), scale)
            except Exception as e:
                print(f"[WARN] Generic Object Detection failed: {e}")

        # 2. Road Damage Detection (Potholes/Cracks/Manholes)
        damages = []
        if self.road_damage_detector.is_loaded:
            try:
                res = self.road_damage_detector.predict(infer_frame)
                damages = self._scale_detections(res.get("detections", []), scale)
            except Exception as e:
                print(f"[WARN] Road Damage Detection failed: {e}")

        # Enrich Potholes with iWatchRoad Severity Analysis
        for dmg in damages:
            if dmg.get("label") == "pothole":
                sev_info = self.pothole_severity_analyzer.analyze(dmg["bbox"], dmg.get("confidence", 1.0))
                dmg["severity_grade"] = sev_info["severity_grade"]
                dmg["area_px"] = sev_info["area_px"]
                dmg["color_bgr"] = sev_info["color_bgr"]

        # 3. Drivable Road Area Segmentation
        seg_mask = None
        drivable_pct = 100.0
        seg_info = {"status": self.road_segmenter.get_status()[0], "drivable_area_percentage": None}

        if self.road_segmenter.is_loaded:
            try:
                res = self.road_segmenter.predict(infer_frame)
                seg_mask = res.get("seg_mask")  # Binary mask (H×W uint8); None if segmenter not loaded
                seg_info = {k: v for k, v in res.items() if k != "seg_mask"}
            except Exception as e:
                print(f"[WARN] Road Segmentation failed: {e}")

        # 4. Traffic Sign Detection (Optional)
        traffic_signs = []
        if self.traffic_sign_detector.is_loaded:
            try:
                res = self.traffic_sign_detector.predict(infer_frame)
                traffic_signs = self._scale_detections(res.get("detections", []), scale)
            except Exception as e:
                print(f"[WARN] Traffic Sign Detection failed: {e}")

        # 5. Spatial Proximity Analysis
        proximity_interactions = self.spatial_analyzer.analyze_proximity(objects, damages, threshold_px=150.0)

        # 6. Road Hazard Indicator Calculation
        hazard_result = self.hazard_calculator.calculate(
            road_users=objects,
            damages=damages,
            drivable_area_pct=drivable_pct,
            proximity_interactions=proximity_interactions
        )

        # 7. Render Visualization Overlay
        annotated_frame = self.visualizer.render(
            frame=image,
            objects=objects,
            damages=damages,
            segmentation_mask=seg_mask,
            traffic_signs=traffic_signs,
            hazard_result=hazard_result,
            proximity_interactions=proximity_interactions
        )

        # 8. Build JSON Report
        json_report = self.json_reporter.generate_report(
            source_name=source_name,
            media_type="image",
            objects=objects,
            damages=damages,
            segmentation_info=seg_info,
            traffic_signs=traffic_signs,
            hazard_result=hazard_result,
            proximity_interactions=proximity_interactions,
            component_statuses=self.get_component_statuses()
        )

        return annotated_frame, json_report

    def _scale_detections(self, detections: list, scale: float) -> list:
        """Scales bounding box coordinates back to original image resolution.

        Args:
            detections: List of detection dicts, each with a 'bbox' key ([x1,y1,x2,y2]).
            scale: The ratio used to downscale the frame before inference (e.g. 0.5).
                   Pass 1.0 to return the list unchanged.

        Returns:
            The same list with bbox coordinates multiplied by 1/scale in-place.
        """
        if scale == 1.0:
            return detections
        inv = 1.0 / scale
        for det in detections:
            det['bbox'] = [c * inv for c in det['bbox']]
        return detections
