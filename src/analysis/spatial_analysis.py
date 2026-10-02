"""
Spatial Analysis Module
Computes spatial proximity and hazard interaction metrics between road users and detected road anomalies.
"""
from typing import List, Dict, Any, Tuple
import math

class SpatialAnalyzer:
    """
    Computes Euclidean distances and overlap ratios between road users and hazards.
    """

    @staticmethod
    def get_center(bbox: list) -> Tuple[float, float]:
        """Returns (cx, cy) center of a [x1, y1, x2, y2] bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def compute_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """Computes Euclidean distance between two points."""
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def analyze_proximity(self, road_users: List[Dict[str, Any]], hazards: List[Dict[str, Any]], threshold_px: float = 150.0) -> List[Dict[str, Any]]:
        """
        Identifies pairs of road users and hazards within threshold_px distance.
        """
        interactions = []
        for user in road_users:
            u_center = self.get_center(user["bbox"])
            for hazard in hazards:
                h_center = self.get_center(hazard["bbox"])
                dist = self.compute_distance(u_center, h_center)
                if dist <= threshold_px:
                    interactions.append({
                        "road_user_label": user.get("label", "user"),
                        "road_user_bbox": user["bbox"],
                        "hazard_label": hazard.get("label", "hazard"),
                        "hazard_bbox": hazard["bbox"],
                        "distance_px": round(dist, 1),
                        "risk_proximity": "CRITICAL" if dist < (threshold_px / 2.0) else "WARNING"
                    })
        return interactions
