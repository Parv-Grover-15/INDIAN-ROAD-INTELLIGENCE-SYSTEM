"""
Visualizer Module
Renders composite annotations on image/video frames:
- Transparent drivable area mask overlay
- Bounding boxes for objects, road damage, and traffic signs
- Severity color coding
- Spatial risk lines
- Road Hazard Indicator HUD banner
"""
import cv2
import numpy as np
from typing import Dict, Any, List

class Visualizer:
    """
    Renders annotations, overlays, and HUD banners on image frames.
    """

    def __init__(self):
        # Color definitions (BGR)
        self.COLOR_DRIVABLE = (57, 219, 57)    # Green overlay
        self.COLOR_VEHICLE = (255, 191, 0)     # Deep Sky Blue / Cyan
        self.COLOR_PEDESTRIAN = (255, 0, 255) # Magenta
        self.COLOR_CRACK = (0, 255, 255)      # Yellow
        self.COLOR_TRAFFIC_LIGHT = (0, 165, 255) # Orange

    def render(
        self,
        frame: np.ndarray,
        objects: List[Dict[str, Any]] = None,
        damages: List[Dict[str, Any]] = None,
        segmentation_mask: np.ndarray = None,
        traffic_signs: List[Dict[str, Any]] = None,
        hazard_result: Dict[str, Any] = None,
        proximity_interactions: List[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Renders composite visual output.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Render Drivable Area Mask Overlay
        if segmentation_mask is not None:
            if segmentation_mask.shape[:2] != (h, w):
                segmentation_mask = cv2.resize(segmentation_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = canvas.copy()
            overlay[segmentation_mask == 1] = self.COLOR_DRIVABLE
            canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

        # 2. Render Proximity Interaction Lines
        if proximity_interactions:
            for inter in proximity_interactions:
                u_box = inter["road_user_bbox"]
                h_box = inter["hazard_bbox"]
                u_center = (int((u_box[0] + u_box[2]) / 2), int((u_box[1] + u_box[3]) / 2))
                h_center = (int((h_box[0] + h_box[2]) / 2), int((h_box[1] + h_box[3]) / 2))
                line_color = (0, 0, 255) if inter["risk_proximity"] == "CRITICAL" else (0, 255, 255)
                cv2.line(canvas, u_center, h_center, line_color, 2, cv2.LINE_AA)

        # 3. Render Generic Objects (Vehicles/Pedestrians)
        if objects:
            for obj in objects:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                label = obj.get("label", "object")
                conf = obj.get("confidence", 0.0)
                color = self.COLOR_PEDESTRIAN if label in ["pedestrian", "person"] else self.COLOR_VEHICLE
                
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                caption = f"{label} {conf:.2f}"
                cv2.putText(canvas, caption, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # 4. Render Road Damage (Potholes / Cracks / Manholes)
        if damages:
            for dmg in damages:
                x1, y1, x2, y2 = map(int, dmg["bbox"])
                label = dmg.get("label", "damage")
                conf = dmg.get("confidence", 0.0)
                color = dmg.get("color_bgr", (0, 0, 255)) if label == "pothole" else self.COLOR_CRACK
                
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                grade = dmg.get("severity_grade", "")
                caption = f"{label} ({grade})" if grade else f"{label} {conf:.2f}"
                cv2.putText(canvas, caption, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # 5. Render Traffic Signs
        if traffic_signs:
            for sign in traffic_signs:
                x1, y1, x2, y2 = map(int, sign["bbox"])
                label = sign.get("label", "sign")
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(canvas, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

        # 6. Render HUD Banner
        if hazard_result:
            score = hazard_result["hazard_score"]
            rating = hazard_result["rating"]
            
            # Draw top semi-transparent banner
            banner_h = 45
            overlay_banner = canvas.copy()
            cv2.rectangle(overlay_banner, (0, 0), (w, banner_h), (20, 20, 20), -1)
            canvas = cv2.addWeighted(canvas, 0.3, overlay_banner, 0.7, 0)

            hud_text = f"INDIAN ROAD INTELLIGENCE | HAZARD INDICATOR: {score}/100 ({rating})"
            cv2.putText(canvas, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas
