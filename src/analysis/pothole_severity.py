"""
Pothole Severity Analyzer
Encapsulates iWatchRoad area-based pothole severity grading logic.
Grade levels: Low Severity, Moderate Severity, High Severity.
"""
from typing import Dict, Any, Tuple

class PotholeSeverityAnalyzer:
    """
    Analyzes pothole bounding boxes and determines severity grade and color code.
    Adapted from iWatchRoad severity logic (code/backend/dashcam_processor/pothole_grading.py).
    """

    @staticmethod
    def grade_pothole(width: float, height: float, confidence: float = 1.0) -> Tuple[str, Tuple[int, int, int]]:
        """
        Grades pothole based on pixel area (width * height).

        Args:
            width (float): Width of bounding box in pixels
            height (float): Height of bounding box in pixels
            confidence (float): Detection confidence

        Returns:
            tuple: (grade_string, (b, g, r) color tuple)
        """
        area = width * height

        if area < 6000:
            grade = "Low Severity"
            color = (0, 255, 0)      # Green
        elif 6000 <= area <= 8000:
            grade = "Moderate Severity"
            color = (0, 165, 255)    # Orange
        else:
            grade = "High Severity"
            color = (0, 0, 255)      # Red

        return grade, color

    def analyze(self, bbox: list, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Analyzes a single bounding box [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        grade, color = self.grade_pothole(w, h, confidence)

        return {
            "bbox": bbox,
            "width": round(w, 1),
            "height": round(h, 1),
            "area_px": round(w * h, 1),
            "severity_grade": grade,
            "color_bgr": color
        }
