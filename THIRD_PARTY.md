# Third-Party GitHub Repositories & Model Attribution

This document details the 5 original third-party GitHub repositories referenced in the **Indian Road Intelligence System** project, their official repository URLs, authors, licenses, datasets, and how they are integrated.

---

## 1. IDD-Indian-Driving-Dataset
* **GitHub Repository URL:** [https://github.com/idd-dataset/IDD-Indian-Driving-Dataset](https://github.com/idd-dataset/IDD-Indian-Driving-Dataset)
* **Author / Organization:** IIIT Hyderabad / AutoNUE
* **License:** Apache License 2.0
* **Model & Dataset:** Multi-label PyTorch ResNet101 tag classifier trained on 9,860 images from IIIT-H IDD.
* **Integration:** Kept completely isolated in `IDD-Indian-Driving-Dataset/`. Used as a reference taxonomy for 16 scene-level tags.

---

## 2. idd-road-segmentation
* **GitHub Repository URL:** [https://github.com/idd-dataset/idd-road-segmentation](https://github.com/idd-dataset/idd-road-segmentation)
* **License:** MIT License
* **Model & Dataset:** ResNet50 U-Net binary drivable-area semantic segmenter trained on IDD Parts I & II (16,063 images).
* **Integration:** Kept completely isolated in `idd-road-segmentation/`. Wrapped via `src/adapters/road_segmenter.py` for drivable surface masking.

---

## 3. Indian-Traffic-Sign-Detection-System
* **GitHub Repository URL:** [https://github.com/akanksha-2002/Indian_traffic_sign_detection_system](https://github.com/akanksha-2002/Indian_traffic_sign_detection_system)
* **License:** Unlicensed (Copyright owned by repository author)
* **Model & Dataset:** YOLOv8s trained on Roboflow Indian Traffic Signboards dataset (56 classes).
* **Integration:** Kept completely isolated in `Indian-Traffic-Sign-Detection-System/`. Wrapped via `src/adapters/traffic_sign.py` as an optional module.

---

## 4. iWatchRoad
* **GitHub Repository URL:** [https://github.com/smlab-niser/iWatchRoad](https://github.com/smlab-niser/iWatchRoad)
* **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
* **Model & Dataset:** Ultralytics YOLOv8 pothole detector + Django web server + area-based pothole grading logic.
* **Integration:** Kept completely isolated in `iWatchRoad/`. Reused area-based pothole severity grading logic (`code/backend/dashcam_processor/pothole_grading.py`) inside `src/analysis/pothole_severity.py`.

---

## 5. RoadScanCV
* **GitHub Repository URL:** [https://github.com/Botkraker/RoadScanCV](https://github.com/Botkraker/RoadScanCV)
* **License:** Unlicensed (Copyright owned by repository author)
* **Model & Dataset:** Ultralytics YOLOv8 multi-class detector (`0: pothole`, `1: crack`, `2: manhole`) trained on 17,732 unified images across 5 public datasets.
* **Integration:** Kept completely isolated in `RoadScanCV/`. Wrapped via `src/adapters/road_damage.py`.
