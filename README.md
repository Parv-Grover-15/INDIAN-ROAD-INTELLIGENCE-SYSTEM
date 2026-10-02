# ROAD//IQ — Indian Road Intelligence System

An integrated computer vision platform for real-time traffic detection, road damage assessment, and safety hazard scoring tailored for unstructured Indian road environments.

---

## ⚡ Quick Start Guide (For Live Presentation)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Parv-Grover-15/INDIAN-ROAD-INTELLIGENCE-SYSTEM.git
cd INDIAN-ROAD-INTELLIGENCE-SYSTEM
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the ROAD//IQ Dashboard
```bash
python -m streamlit run app.py
```

The application will immediately launch in your browser at `http://localhost:8501`.

---

## 🌟 Key Capabilities & Features

* **Multi-Class Road User Detection (YOLOv8):** Vehicles, motorcycles, auto-rickshaws, buses, trucks, pedestrians, and traffic lights.
* **Road Damage & Severity Assessment:** Detection of potholes, structural cracks, and manholes with area-based severity grading (`Low Severity`, `Moderate Severity`, `High Severity`).
* **Drivable Area Segmentation (IDD U-Net):** Pixel-level semantic segmentation trained on IIIT Hyderabad Indian Driving Dataset (IDD).
* **Spatial Risk Vector Analysis:** Real-time Euclidean distance calculation between road users and road damage anomalies.
* **Road Hazard Indicator (0–100):** Transparent 0–100 score and categorical hazard ratings (`LOW`, `MODERATE`, `HIGH`, `VERY HIGH`).
* **Frame-by-Frame Video Analytics:** Web-compatible H.264 video rendering with single-loading model performance.
* **Technical Reports:** Instant 1-click JSON report generation and history tracking.

---

## 📊 Project Architecture

```
                       [ INPUT IMAGE / VIDEO ]
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  UNIFIED PIPELINE ENGINE    │
                   └──────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Generic Objects  │    │   Road Damage    │    │ Drivable Area    │
│ (YOLOv8 COCO)    │    │ (Potholes/Cracks)│    │ (IDD U-Net)      │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │ OUR INTEGRATION LAYER    │
                     │ • Spatial Proximity      │
                     │ • Pothole Severity       │
                     │ • Road Hazard Indicator  │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  STREAMLIT DASHBOARD UI   │
                    │   Annotated Output + JSON │
                    └───────────────────────────┘
```

---

## 📁 Repository Structure

```
INDIAN-ROAD-INTELLIGENCE-SYSTEM/
├── app.py                      # Main Streamlit Dashboard UI
├── requirements.txt            # Python Dependency Manifest
├── README.md                   # Project Overview & Presentation Guide
├── THIRD_PARTY.md              # Open-Source Repository Attribution
├── yolov8n.pt                  # Pre-trained Object Detection Weights
│
├── src/                        # Core Application Software Layer
│   ├── adapters/               # Decoupled Model Adapter Wrappers
│   ├── analysis/               # Spatial Proximity & Hazard Score Math
│   └── pipeline/               # Image & Video Processing Engines
│
├── docs/                       # Documentation & Presentation Artifacts
│   ├── Indian_Road_Intelligence_System_Presentation.pptx
│   ├── presentation.html
│   ├── verification.md
│   └── our_contribution.md
│
└── weights/                    # Model Checkpoint Directory
```

---

## 🔗 Integrated Third-Party Repositories

1. [IDD-Indian-Driving-Dataset](https://github.com/idd-dataset/IDD-Indian-Driving-Dataset) (IIIT Hyderabad)
2. [idd-road-segmentation](https://github.com/idd-dataset/idd-road-segmentation)
3. [Indian-Traffic-Sign-Detection-System](https://github.com/akanksha-2002/Indian_traffic_sign_detection_system)
4. [iWatchRoad](https://github.com/smlab-niser/iWatchRoad)
5. [RoadScanCV](https://github.com/Botkraker/RoadScanCV)
