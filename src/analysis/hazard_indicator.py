"""
Road Hazard Indicator Calculator
Calculates a 0-100 contextual hazard score based on an engineering heuristic.

NOTE: The Road Hazard Indicator is an engineering heuristic combining hazard severity,
traffic volume, pedestrian presence, and spatial proximity. It is NOT a scientifically
validated accident prediction model.
"""
from typing import Dict, Any, List

class RoadHazardIndicatorCalculator:
    """
    Computes transparent engineering heuristic score (0-100) for Indian road scenes.
    """

    def calculate(
        self,
        road_users: List[Dict[str, Any]],
        damages: List[Dict[str, Any]],
        drivable_area_pct: float = 100.0,
        proximity_interactions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates score (0-100) and severity rating.
        """
        if proximity_interactions is None:
            proximity_interactions = []

        base_score = 0.0

        # 1. Potholes & Damage Penalties
        potholes = [d for d in damages if d.get("label") == "pothole"]
        cracks = [d for d in damages if d.get("label") == "crack"]
        manholes = [d for d in damages if d.get("label") == "manhole"]

        # Base count penalty
        base_score += len(potholes) * 12.0
        base_score += len(cracks) * 8.0
        base_score += len(manholes) * 5.0

        # Pothole Severity Penalties
        high_sev_count = 0
        mod_sev_count = 0
        low_sev_count = 0

        for p in potholes:
            grade = p.get("severity_grade", "Low Severity")
            if grade == "High Severity":
                base_score += 15.0
                high_sev_count += 1
            elif grade == "Moderate Severity":
                base_score += 8.0
                mod_sev_count += 1
            else:
                base_score += 3.0
                low_sev_count += 1

        # 2. Road User Vulnerability
        pedestrians = [u for u in road_users if u.get("label") in ["pedestrian", "person"]]
        vehicles = [u for u in road_users if u.get("label") in ["car", "motorcycle", "bicycle", "bus", "truck"]]

        base_score += len(pedestrians) * 7.0
        base_score += len(vehicles) * 2.0

        # 3. Spatial Proximity Multiplier
        critical_interactions = [i for i in proximity_interactions if i.get("risk_proximity") == "CRITICAL"]
        base_score += len(critical_interactions) * 10.0
        base_score += (len(proximity_interactions) - len(critical_interactions)) * 5.0

        # 4. Reduced Drivable Area Penalty
        if drivable_area_pct < 50.0 and drivable_area_pct > 0.0:
            base_score += (50.0 - drivable_area_pct) * 0.5

        # Clamp score between 0 and 100
        final_score = int(min(100.0, max(0.0, round(base_score))))

        # Rating Category
        if final_score < 25:
            rating = "LOW"
            color_hex = "#28a745" # Green
        elif final_score < 55:
            rating = "MODERATE"
            color_hex = "#ffc107" # Yellow
        elif final_score < 80:
            rating = "HIGH"
            color_hex = "#fd7e14" # Orange
        else:
            rating = "VERY HIGH"
            color_hex = "#dc3545" # Red

        return {
            "hazard_score": final_score,
            "rating": rating,
            "color_hex": color_hex,
            "disclaimer": "Engineering heuristic evaluation based on road users, hazards, and spatial proximity. Not an accident prediction model.",
            "metrics_breakdown": {
                "pedestrians_count": len(pedestrians),
                "vehicles_count": len(vehicles),
                "total_potholes": len(potholes),
                "high_severity_potholes": high_sev_count,
                "moderate_severity_potholes": mod_sev_count,
                "low_severity_potholes": low_sev_count,
                "total_cracks": len(cracks),
                "spatial_proximity_alerts": len(proximity_interactions),
                "drivable_area_percentage": round(drivable_area_pct, 1)
            }
        }
