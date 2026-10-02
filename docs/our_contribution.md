# Our Contribution - Indian Road Intelligence System

This document clearly demarcates the original software contributions developed for the **Indian Road Intelligence System** from third-party repositories.

---

## 1. Scope of Our Contribution

While five third-party repositories were inspected and integrated as isolated reference modules, our team developed the unified application software layer (`src/`, `app.py`, and `scratch/`).

### Key Original Contributions:

1. **Decoupled Model Adapter Architecture (`src/adapters/`):**
   * Implemented a unified `BaseModelAdapter` interface ensuring third-party ML repositories remain untouched.
   * Built robust fallback error handling so that if any model weights file is missing, the application gracefully disables that module without crashing the system.

2. **Indian Road Contextual Analysis Engine (`src/analysis/`):**
   * **`pothole_severity.py`:** Standardized size-based severity grading (`Low Severity` < 6000 px, `Moderate Severity` 6000-8000 px, `High Severity` > 8000 px).
   * **`spatial_analysis.py`:** Euclidean proximity calculations determining risk interactions between road users (pedestrians, vehicles) and road damage anomalies.
   * **`hazard_indicator.py`:** Formulated a transparent **Road Hazard Indicator (0–100)** score and categorical ratings (`LOW`, `MODERATE`, `HIGH`, `VERY HIGH`).

3. **Unified Multi-Model Execution Pipelines (`src/pipeline/`):**
   * **`image_pipeline.py`:** Processes single image frames, combines detections from all active models, performs spatial analysis, and computes hazard metrics.
   * **`video_pipeline.py`:** Processes video clips while loading models **EXACTLY ONCE** at initialization to preserve performance.

4. **Visualization & Reporting Engine (`src/pipeline/visualizer.py` & `json_reporter.py`):**
   * Renders color-coded bounding boxes, severity badges, transparent drivable area mask overlays, spatial risk vectors, and a top-level HUD hazard banner.
   * Produces standardized JSON reports detailing detection counts, damage classifications, and component statuses.

5. **Streamlit Interactive UI Dashboard (`app.py`):**
   * Created a clean web interface supporting image/video uploads, real-time model status monitoring, statistics dashboards, and file downloads.
