"""
JSON Reporter Module
Formats comprehensive JSON analysis reports for image and video inference runs.
"""
from typing import Dict, Any, List

class JSONReporter:
    """
    Generates standardized JSON report structures.
    """

    def generate_report(
        self,
        source_name: str,
        media_type: str,
        objects: List[Dict[str, Any]] = None,
        damages: List[Dict[str, Any]] = None,
        segmentation_info: Dict[str, Any] = None,
        traffic_signs: List[Dict[str, Any]] = None,
        hazard_result: Dict[str, Any] = None,
        proximity_interactions: List[Dict[str, Any]] = None,
        component_statuses: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Builds report dictionary.
        """
        if objects is None: objects = []
        if damages is None: damages = []
        if traffic_signs is None: traffic_signs = []
        if proximity_interactions is None: proximity_interactions = []
        if component_statuses is None: component_statuses = {}

        # Class breakdown
        object_counts = {}
        for obj in objects:
            lbl = obj.get("label", "unknown")
            object_counts[lbl] = object_counts.get(lbl, 0) + 1

        damage_counts = {}
        for dmg in damages:
            lbl = dmg.get("label", "unknown")
            damage_counts[lbl] = damage_counts.get(lbl, 0) + 1

        report = {
            "project": "Indian Road Intelligence System",
            "source_media": source_name,
            "media_type": media_type,
            "summary": {
                "total_road_users_detected": len(objects),
                "total_damage_anomalies_detected": len(damages),
                "total_traffic_signs_detected": len(traffic_signs),
                "road_hazard_indicator": hazard_result.get("hazard_score") if hazard_result else None,
                "hazard_rating": hazard_result.get("rating") if hazard_result else None,
                "disclaimer": hazard_result.get("disclaimer") if hazard_result else ""
            },
            "component_statuses": component_statuses,
            "counts_breakdown": {
                "road_users": object_counts,
                "road_damage": damage_counts
            },
            "road_hazard_indicator_details": hazard_result,
            "road_segmentation": segmentation_info or {"status": "UNAVAILABLE"},
            "spatial_proximity_alerts": proximity_interactions,
            "detailed_detections": {
                "road_users": objects,
                "road_damage": damages,
                "traffic_signs": traffic_signs
            }
        }
        return self._make_serializable(report)

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() if k != "seg_mask"}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(v) for v in obj]
        elif hasattr(obj, "item") and callable(getattr(obj, "item")):
            return obj.item()
        elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
            return obj.tolist()
        return obj
