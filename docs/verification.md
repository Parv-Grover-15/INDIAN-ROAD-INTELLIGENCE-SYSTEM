# Component Verification Report

Generated automatically by `scratch/verify_components.py`.

## Component Status Table

| Component | Model | Weights | Inference | CPU | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Generic Object Detector | YOLOv8 (COCO Pre-trained) | yolov8n.pt (COCO Pre-trained) | SUCCESS | Supported | **PASS** |
| Road Damage Detector | RoadScanCV YOLO (Pothole/Crack/Manhole) | weights/road_damage_best.pt | NOT RUN | Supported | `FAIL` |
| Road Segmentation | IDD U-Net (ResNet50 Encoder) | weights/road_segmentation_best.pt | NOT RUN | Supported | `FAIL` |
| Pothole Severity Logic | iWatchRoad Area Rule Heuristic | N/A (Code Logic) | SUCCESS | Supported | **PASS** |
| Traffic Sign Detector (Optional) | Indian Traffic Sign Detector (YOLOv8s 56-class) | weights/traffic_signs_best.pt | NOT RUN | Supported | `WARN` |

## Detailed Status Messages

- **Generic Object Detector**: `[PASS]` Component loaded successfully (yolov8n.pt)
- **Road Damage Detector**: `[FAIL]` Missing weights: Pretrained checkpoint not found at 'weights/road_damage_best.pt'
- **Road Segmentation**: `[FAIL]` Missing weights: Pretrained checkpoint not found at 'weights/road_segmentation_best.pt'
- **Pothole Severity Logic**: `[PASS]` Component loaded & verified successfully
- **Traffic Sign Detector (Optional)**: `[WARN]` Optional component unavailable: Checkpoint not found at 'weights/traffic_signs_best.pt'
