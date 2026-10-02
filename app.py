"""
ROAD//IQ - Indian Road Intelligence System
Streamlit Frontend & Full-Stack Application Interface
"""
import os
import json
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import ImagePipeline, VideoPipeline

# Page Config
st.set_page_config(
    page_title="ROAD//IQ - Indian Road Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching the mockup theme (#121316 dark theme with #f59e0b Orange/Gold accents)
MOCKUP_CSS = """
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #121316;
        color: #e4e5e7;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #18191d;
        border-right: 1px solid #2c2d32;
    }
    .brand-title {
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .brand-title span {
        color: #f59e0b;
    }
    .brand-subtitle {
        font-size: 0.85rem;
        color: #8a8d93;
        margin-bottom: 1.5rem;
    }
    
    /* System Status Card in Sidebar */
    .sys-status-card {
        background-color: #202126;
        border: 1px solid #2c2d32;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
    }
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
    }

    /* Top Header Bar */
    .header-tag {
        color: #f59e0b;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.2rem;
        margin-bottom: 0.3rem;
    }
    .header-desc {
        color: #909399;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Uploader Container Card (Off-White in Mockup) */
    .uploader-card {
        background-color: #f5f5f0;
        color: #1a1a1a;
        border-radius: 12px;
        border-left: 5px solid #f59e0b;
        padding: 1.5rem 2rem;
        margin-bottom: 1.8rem;
    }
    .uploader-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .uploader-sub {
        font-size: 0.85rem;
        color: #666666;
    }

    /* Content Cards */
    .content-card {
        background-color: #1c1d22;
        border: 1px solid #2c2d32;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .card-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #8a8d93;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* Hazard Indicator Card */
    .hazard-score-text {
        font-size: 3.2rem;
        font-weight: 900;
        color: #f59e0b;
        line-height: 1;
    }
    .hazard-denom {
        font-size: 1.2rem;
        color: #71747a;
        font-weight: 500;
    }
    .risk-badge {
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .risk-low { color: #10b981; }
    .risk-moderate { color: #f59e0b; }
    .risk-high { color: #f97316; }
    .risk-very-high { color: #ef4444; }

    /* Custom Buttons - Orange Primary Accent */
    div.stButton > button {
        background-color: #f59e0b !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #fbbf24 !important;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4) !important;
    }

    /* Metric Counter Blocks */
    .metric-block {
        background-color: #24252b;
        border: 1px solid #2e3037;
        border-radius: 10px;
        padding: 1rem;
        text-align: left;
    }
    .metric-block-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-block-lbl {
        font-size: 0.8rem;
        color: #8a8d93;
        margin-top: 0.2rem;
    }

    /* Detection Summary Row */
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #2a2b30;
        font-size: 0.9rem;
    }
    .summary-row:last-child {
        border-bottom: none;
    }
    .summary-name { color: #d1d5db; font-weight: 500; }
    .summary-val { color: #f59e0b; font-weight: 700; }
</style>
"""

st.markdown(MOCKUP_CSS, unsafe_allow_html=True)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'reset_key' not in st.session_state:
    st.session_state['reset_key'] = 0
if 'video_result' not in st.session_state:
    st.session_state['video_result'] = None

# Initialize Models ONCE
@st.cache_resource(show_spinner="Loading ROAD//IQ Engine & Model Adapters...")
def load_models():
    return ImagePipeline(), VideoPipeline()

image_pipeline, video_pipeline = load_models()
statuses = image_pipeline.get_component_statuses()

# Sidebar Navigation matching Mockup
with st.sidebar:
    st.markdown('<div class="brand-title">ROAD<span>//IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Indian Road Intelligence</div>', unsafe_allow_html=True)

    nav_option = st.radio(
        "NAVIGATION",
        ["Overview", "Analyze Image", "Analyze Video", "History", "Model Status", "About Project", "GitHub / Sources"],
        index=0,
        label_visibility="collapsed"
    )

    # Bottom System Status Card
    active_count = sum(1 for s in statuses.values() if s == "PASS")
    st.markdown(f"""
    <div class="sys-status-card">
        <div style="font-size: 0.75rem; color: #8a8d93; font-weight: 700; text-transform: uppercase;">SYSTEM STATUS</div>
        <div style="margin-top: 0.4rem; font-size: 0.95rem; font-weight: 700; color: #ffffff;">
            <span class="status-dot"></span> Pipeline online
        </div>
        <div style="font-size: 0.8rem; color: #909399; margin-top: 0.2rem;">{active_count} / {len(statuses)} components active</div>
    </div>
    """, unsafe_allow_html=True)

# Callback for Reset Button "+ New analysis"
def reset_analysis():
    st.session_state['reset_key'] += 1
    st.session_state['video_result'] = None
    st.rerun()

# Header Section
col_head, col_btn = st.columns([4, 1])
with col_head:
    st.markdown('<div class="header-tag">ROAD SCENE ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-title">Indian Road Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-desc">Upload a road image or video. Detect traffic, road hazards, damages, and scene conditions.</div>', unsafe_allow_html=True)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("＋ New analysis", on_click=reset_analysis, use_container_width=True)

# -------------------------------------------------------------------
# VIEW 1: OVERVIEW & IMAGE ANALYSIS
# -------------------------------------------------------------------
if nav_option in ["Overview", "Analyze Image"]:
    
    # Drag & Drop Uploader Container
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state['reset_key']}",
        help="JPG or PNG up to 200 MB"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="uploader-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="uploader-title">Drop a road image or video</div>
                    <div class="uploader-sub">JPG, PNG, MP4 or MOV · up to 200 MB</div>
                </div>
                <div style="text-align: right; color: #666; font-size: 0.8rem; font-weight: 600;">
                    IMAGE / VIDEO<br><span style="color: #999;">Analysis starts after upload</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👆 Upload an image above to display the analysis preview and hazard metrics.")

    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Executing neural inference across models..."):
            annotated_bgr, json_report = image_pipeline.process(image_bgr, source_name=uploaded_file.name)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        summary = json_report.get("summary", {})
        hazard_score = summary.get("road_hazard_indicator", 0)
        hazard_rating = summary.get("hazard_rating", "LOW")
        
        history_item = {
            "file_name": uploaded_file.name,
            "time": time.strftime("%H:%M:%S"),
            "objects": summary.get("total_road_users_detected", 0),
            "damage": summary.get("total_damage_anomalies_detected", 0),
            "hazard_score": hazard_score,
            "rating": hazard_rating
        }
        if history_item not in st.session_state['history']:
            st.session_state['history'].append(history_item)

        left_col, right_col = st.columns([1.8, 1])

        with left_col:
            st.markdown('<div class="card-label">ANALYSIS PREVIEW</div>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)
            
            obj_cnt = summary.get("total_road_users_detected", 0)
            dmg_cnt = summary.get("total_damage_anomalies_detected", 0)
            
            st.markdown(f"""
            <div style="background-color: #18191d; border: 1px solid #2c2d32; border-radius: 8px; padding: 0.8rem 1.2rem; display: flex; justify-content: space-between; font-size: 0.85rem;">
                <div><strong>DETECTED:</strong> <span style="color: #f59e0b; font-weight: 700;">{obj_cnt} objects</span></div>
                <div>Road damage: <span style="color: #ffffff;">{dmg_cnt}</span></div>
                <div style="color: #8a8d93;">Model: YOLOv8 + U-Net</div>
            </div>
            """, unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="card-label">ROAD HAZARD INDICATOR</div>', unsafe_allow_html=True)
            risk_class = f"risk-{hazard_rating.lower().replace(' ', '-')}"
            
            st.markdown(f"""
            <div class="content-card">
                <div class="hazard-score-text">{hazard_score} <span class="hazard-denom">/ 100</span></div>
                <div class="risk-badge {risk_class}">● {hazard_rating} CURRENT RISK</div>
                <div style="background-color: #2a2b30; border-radius: 4px; height: 8px; width: 100%; margin-bottom: 0.8rem;">
                    <div style="background-color: #f59e0b; height: 8px; border-radius: 4px; width: {hazard_score}%;"></div>
                </div>
                <div style="font-size: 0.75rem; color: #8a8d93;">Heuristic based on detected traffic + hazards + spatial risk</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="card-label">DETECTION SUMMARY</div>', unsafe_allow_html=True)
            counts = json_report.get("counts_breakdown", {}).get("road_users", {})
            cars_cnt = counts.get("car", 0)
            moto_cnt = counts.get("motorcycle", 0)
            people_cnt = counts.get("person", 0) + counts.get("pedestrian", 0)
            heavy_cnt = counts.get("bus", 0) + counts.get("truck", 0)

            st.markdown(f"""
            <div class="content-card">
                <div class="summary-row">
                    <span class="summary-name">Cars</span>
                    <span class="summary-val">{cars_cnt:02d}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-name">Motorcycles</span>
                    <span class="summary-val">{moto_cnt:02d}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-name">People</span>
                    <span class="summary-val">{people_cnt:02d}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-name">Bus / Truck</span>
                    <span class="summary-val">{heavy_cnt:02d}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">ANALYSIS OUTPUT</div>', unsafe_allow_html=True)

        m_col1, m_col2, m_col3, m_col4, m_btn = st.columns([1, 1, 1, 1, 1.2])

        with m_col1:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-block-val">{obj_cnt}</div>
                <div class="metric-block-lbl">Objects</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-block-val">{dmg_cnt}</div>
                <div class="metric-block-lbl">Road damage</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            signs_cnt = summary.get("total_traffic_signs_detected", 0)
            signs_str = f"{signs_cnt}" if statuses.get("traffic_sign_detector") == "PASS" else "—"
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-block-val">{signs_str}</div>
                <div class="metric-block-lbl">Signs</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            drivable = json_report.get('road_segmentation', {}).get('drivable_area_percentage')
            drivable_str = f"{drivable}%" if drivable is not None else "—"
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-block-val">{drivable_str}</div>
                <div class="metric-block-lbl">Segmentation</div>
            </div>
            """, unsafe_allow_html=True)

        with m_btn:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            json_bytes = json.dumps(json_report, indent=2).encode("utf-8")
            st.download_button(
                label="Export JSON",
                data=json_bytes,
                file_name=f"report_{uploaded_file.name}.json",
                mime="application/json",
                use_container_width=True
            )

# -------------------------------------------------------------------
# VIEW 2: ANALYZE VIDEO
# -------------------------------------------------------------------
elif nav_option == "Analyze Video":
    st.subheader("🎥 Video Analysis Pipeline")
    uploaded_video = st.file_uploader("Upload an Indian Road Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        st.subheader("Source Input Video")
        st.video(tfile.name)

        if st.button("🚀 Process Video Frames", use_container_width=True):
            with st.spinner("Processing video frames & rendering web H.264 video (Models loaded ONCE)..."):
                out_path = os.path.join(tempfile.gettempdir(), f"analyzed_{uploaded_video.name}.mp4")
                res_path, video_report = video_pipeline.process_video(
                    video_input_path=tfile.name,
                    output_video_path=out_path,
                    frame_stride=2
                )
                
                # Read output video bytes into memory for persistent display & download
                with open(res_path, "rb") as vf:
                    video_bytes = vf.read()

                st.session_state['video_result'] = {
                    "video_bytes": video_bytes,
                    "file_name": f"analyzed_{uploaded_video.name}.mp4",
                    "report": video_report
                }

        # Render Video Results if present in session state
        if st.session_state['video_result'] is not None:
            v_res = st.session_state['video_result']
            video_bytes = v_res["video_bytes"]
            video_report = v_res["report"]

            st.success("✅ Video Analysis Complete!")
            st.subheader("Annotated Output Video")
            st.video(video_bytes)

            v_haz = video_report.get("video_hazard_summary", {})
            avg_h = v_haz.get("average_road_hazard_indicator", 0)
            peak_h = v_haz.get("peak_road_hazard_indicator", 0)
            rating = v_haz.get("overall_hazard_rating", "N/A")
            rating_class = f"risk-{rating.lower().replace(' ', '-')}"

            st.markdown("---")
            st.subheader("📊 Video Hazard Assessment Summary")
            
            st.markdown(f"""
            <div class="content-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #8a8d93; font-size: 0.85rem; text-transform: uppercase;">Average Hazard Score</div>
                        <div class="hazard-score-text">{avg_h} <span class="hazard-denom">/ 100</span></div>
                    </div>
                    <div>
                        <div style="color: #8a8d93; font-size: 0.85rem; text-transform: uppercase;">Peak Hazard Score</div>
                        <div style="font-size: 2.2rem; font-weight: 700; color: #ffffff;">{peak_h}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #8a8d93; font-size: 0.85rem; text-transform: uppercase;">Overall Rating</div>
                        <div style="font-size: 1.8rem; font-weight: 800;" class="{rating_class}">● {rating}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📥 Export Video Results")
            dl1, dl2 = st.columns(2)

            dl1.download_button(
                label="Download Analyzed Video (.mp4)",
                data=video_bytes,
                file_name=v_res["file_name"],
                mime="video/mp4",
                use_container_width=True
            )

            json_str = json.dumps(video_report, indent=2)
            dl2.download_button(
                label="Download Video Technical Report (.json)",
                data=json_str,
                file_name=f"video_report_{uploaded_video.name}.json",
                mime="application/json",
                use_container_width=True
            )

            with st.expander("🔍 View Video Technical Report JSON"):
                st.json(video_report)

        try: os.remove(tfile.name)
        except Exception: pass

# -------------------------------------------------------------------
# VIEW 3: HISTORY
# -------------------------------------------------------------------
elif nav_option == "History":
    st.subheader("📋 Session Analysis History")
    
    if not st.session_state['history']:
        st.info("No analyses recorded in this session yet. Upload an image or video to populate history.")
    else:
        for idx, item in enumerate(reversed(st.session_state['history'])):
            st.markdown(f"""
            <div class="content-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #ffffff; font-size: 1.1rem;">{item['file_name']}</strong>
                        <div style="font-size: 0.8rem; color: #8a8d93;">Time: {item['time']} | Objects: {item['objects']} | Damages: {item['damage']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.4rem; font-weight: 800; color: #f59e0b;">{item['hazard_score']} / 100</div>
                        <div style="font-size: 0.8rem; font-weight: 700;" class="risk-{item['rating'].lower()}">{item['rating']} RISK</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# VIEW 4: MODEL STATUS
# -------------------------------------------------------------------
elif nav_option == "Model Status":
    st.subheader("⚙️ Neural Component & Model Status")
    
    col_a, col_b = st.columns(2)
    for idx, (comp, status) in enumerate(statuses.items()):
        name = comp.replace("_", " ").title()
        target_col = col_a if idx % 2 == 0 else col_b
        
        with target_col:
            st.markdown(f"""
            <div class="content-card">
                <div style="font-size: 0.8rem; color: #8a8d93; text-transform: uppercase;">MODEL MODULE</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff;">{name}</div>
                <div style="margin-top: 0.5rem;">Status: <strong>{status}</strong></div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# VIEW 5: ABOUT PROJECT
# -------------------------------------------------------------------
elif nav_option == "About Project":
    st.subheader("ℹ️ About Indian Road Intelligence System")
    st.markdown("""
    **INDIAN ROAD INTELLIGENCE SYSTEM** is an integrated computer vision architecture designed specifically for Indian road scenes.
    
    ### Key Features:
    - **Multi-Class Road User Detection:** Vehicles, motorcycles, auto-rickshaws, buses, trucks, and pedestrians.
    - **Road Damage Analysis:** Detection of potholes, cracks, and manholes with iWatchRoad size-based severity grading.
    - **Drivable Area Segmentation:** Pixel-level U-Net segmentation trained on IIIT Hyderabad Indian Driving Dataset (IDD).
    - **Spatial Risk & Proximity Engine:** Computes risk interaction vectors between hazards and vulnerable road users.
    - **Road Hazard Indicator (0–100):** Transparent engineering heuristic score evaluating real-time road risk.
    """)

# -------------------------------------------------------------------
# VIEW 6: GITHUB / SOURCES
# -------------------------------------------------------------------
elif nav_option == "GitHub / Sources":
    st.subheader("🔗 GitHub Repositories & Source Attribution")
    st.markdown("All 5 underlying open-source repositories integrated into the system:")

    sources = [
        {
            "name": "IDD-Indian-Driving-Dataset",
            "url": "https://github.com/idd-dataset/IDD-Indian-Driving-Dataset",
            "task": "Multi-Label Scene Classification",
            "license": "Apache License 2.0",
            "author": "IIIT Hyderabad / AutoNUE"
        },
        {
            "name": "idd-road-segmentation",
            "url": "https://github.com/idd-dataset/idd-road-segmentation",
            "task": "Drivable Road Surface Segmentation (ResNet50 U-Net)",
            "license": "MIT License",
            "author": "IDD Community"
        },
        {
            "name": "Indian-Traffic-Sign-Detection-System",
            "url": "https://github.com/akanksha-2002/Indian_traffic_sign_detection_system",
            "task": "Indian Traffic Signboard Detection (YOLOv8s)",
            "license": "Unlicensed",
            "author": "Akanksha"
        },
        {
            "name": "iWatchRoad",
            "url": "https://github.com/smlab-niser/iWatchRoad",
            "task": "Pothole Tracking & Severity Grading Logic",
            "license": "CC BY-NC-SA 4.0",
            "author": "SMLab NISER"
        },
        {
            "name": "RoadScanCV",
            "url": "https://github.com/Botkraker/RoadScanCV",
            "task": "Road Damage Object Detection (Potholes, Cracks, Manholes)",
            "license": "Unlicensed",
            "author": "Botkraker"
        }
    ]

    for s in sources:
        st.markdown(f"""
        <div class="content-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f59e0b;">{s['name']}</div>
                    <div style="font-size: 0.85rem; color: #909399;">Task: {s['task']} | Author: {s['author']}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">License: {s['license']}</div>
                </div>
                <div>
                    <a href="{s['url']}" target="_blank" style="background-color: #2c2d32; color: #ffffff; padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; font-size: 0.85rem; font-weight: 600;">View GitHub Repository ↗</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
