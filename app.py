"""
ROAD//IQ — Indian Road Intelligence System
Modern Developer UI: Collapsible SVG Icon Sidebar + Clean Main Screen
"""
import os
import json
import time
import tempfile
import cv2
import numpy as np
import streamlit as st

from src.pipeline import ImagePipeline, VideoPipeline

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ROAD//IQ · Indian Road Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Parv-Grover-15/INDIAN-ROAD-INTELLIGENCE-SYSTEM",
        "About": "ROAD//IQ — AI-powered Indian Road Intelligence System"
    }
)

# ─── CSS Theme ────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ─── Keyframes ─── */
@keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeUp   { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulseDot { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }

/* ─── Base Reset ─── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  background: #f8fafc !important;
  color: #0f172a !important;
}
#MainMenu, footer, header { visibility: hidden !important; height: 0 !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Main content width & padding */
.block-container {
  padding: 1.2rem 2.2rem 3.5rem !important;
  max-width: 1380px !important;
}

/* ══════════════════════════════════════════════════════════════════════════════
   SIDEBAR — Fixed width (230px) with SVG icons + text labels
══════════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: #0f172a !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
  width: 230px !important;
  min-width: 230px !important;
  box-shadow: 2px 0 12px rgba(0,0,0,0.05);
  z-index: 100 !important;
}
section[data-testid="stSidebar"] .block-container {
  padding: 1.2rem 0.8rem 2rem !important;
}

/* Sidebar Logo Header */
.sb-logo-box {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0.4rem 0.6rem 0.8rem;
  margin-bottom: 0.8rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  white-space: nowrap;
}
.sb-logo-mark {
  width: 36px; height: 36px;
  border-radius: 9px;
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  color: #ffffff;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 1.05rem;
  flex-shrink: 0;
}
.sb-logo-name {
  font-size: 1.05rem;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: -0.3px;
  opacity: 1;
}

/* Radio Navigation Styling */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] {
  display: flex !important;
  flex-direction: column !important;
  gap: 6px !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] {
  padding: 0.65rem 0.8rem !important;
  border-radius: 10px !important;
  border: 1px solid transparent !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  white-space: nowrap !important;
  transition: background 0.18s ease, border-color 0.18s ease !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
  background: rgba(255,255,255,0.06) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {
  background: rgba(245,158,11,0.15) !important;
  border-color: rgba(245,158,11,0.3) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] p {
  color: #f59e0b !important;
  font-weight: 600 !important;
}

/* Hide standard radio circle input */
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
  display: none !important;
}

/* Text Container: Always visible next to SVG icon */
div[data-testid="stRadio"] label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] {
  display: block !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] p {
  color: #94a3b8 !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  margin: 0 !important;
  white-space: nowrap !important;
}

/* SVG Vector Icons via ::before pseudo-elements */
div[data-testid="stRadio"] label[data-baseweb="radio"]::before {
  content: '';
  width: 20px;
  height: 20px;
  flex-shrink: 0;
  display: inline-block;
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  margin-right: 12px;
}

/* 1. Overview */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='14' width='7' height='7'%3E%3C/rect%3E%3Crect x='3' y='14' width='7' height='7'%3E%3C/rect%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='14' width='7' height='7'%3E%3C/rect%3E%3Crect x='3' y='14' width='7' height='7'%3E%3C/rect%3E%3C/svg%3E");
}

/* 2. Analyze Image */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2' ry='2'%3E%3C/rect%3E%3Ccircle cx='8.5' cy='8.5' r='1.5'%3E%3C/circle%3E%3Cpolyline points='21 15 16 10 5 21'%3E%3C/polyline%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2' ry='2'%3E%3C/rect%3E%3Ccircle cx='8.5' cy='8.5' r='1.5'%3E%3C/circle%3E%3Cpolyline points='21 15 16 10 5 21'%3E%3C/polyline%3E%3C/svg%3E");
}

/* 3. Analyze Video */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='2' width='20' height='20' rx='2.18' ry='2.18'%3E%3C/rect%3E%3Cline x1='7' y1='2' x2='7' y2='22'%3E%3C/line%3E%3Cline x1='17' y1='2' x2='17' y2='22'%3E%3C/line%3E%3Cline x1='2' y1='12' x2='22' y2='12'%3E%3C/line%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='2' width='20' height='20' rx='2.18' ry='2.18'%3E%3C/rect%3E%3Cline x1='7' y1='2' x2='7' y2='22'%3E%3C/line%3E%3Cline x1='17' y1='2' x2='17' y2='22'%3E%3C/line%3E%3Cline x1='2' y1='12' x2='22' y2='12'%3E%3C/line%3E%3C/svg%3E");
}

/* 4. Live Camera */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(4)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M23 7l-7 5 7 5V7z'%3E%3C/path%3E%3Crect x='1' y='5' width='15' height='14' rx='2' ry='2'%3E%3C/rect%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(4)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M23 7l-7 5 7 5V7z'%3E%3C/path%3E%3Crect x='1' y='5' width='15' height='14' rx='2' ry='2'%3E%3C/rect%3E%3C/svg%3E");
}

/* 5. History */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(5)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Cpolyline points='12 6 12 12 16 14'%3E%3C/polyline%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(5)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Cpolyline points='12 6 12 12 16 14'%3E%3C/polyline%3E%3C/svg%3E");
}

/* 6. Model Status */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(6)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='4' width='16' height='16' rx='2' ry='2'%3E%3C/rect%3E%3Crect x='9' y='9' width='6' height='6'%3E%3C/rect%3E%3Cline x1='9' y1='1' x2='9' y2='4'%3E%3C/line%3E%3Cline x1='15' y1='1' x2='15' y2='4'%3E%3C/line%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(6)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='4' width='16' height='16' rx='2' ry='2'%3E%3C/rect%3E%3Crect x='9' y='9' width='6' height='6'%3E%3C/rect%3E%3Cline x1='9' y1='1' x2='9' y2='4'%3E%3C/line%3E%3Cline x1='15' y1='1' x2='15' y2='4'%3E%3C/line%3E%3C/svg%3E");
}

/* 7. About */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(7)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Cline x1='12' y1='16' x2='12' y2='12'%3E%3C/line%3E%3Cline x1='12' y1='8' x2='12.01' y2='8'%3E%3C/line%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(7)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Cline x1='12' y1='16' x2='12' y2='12'%3E%3C/line%3E%3Cline x1='12' y1='8' x2='12.01' y2='8'%3E%3C/line%3E%3C/svg%3E");
}

/* 8. Sources */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(8)::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71'%3E%3C/path%3E%3Cpath d='M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71'%3E%3C/path%3E%3C/svg%3E");
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(8)[aria-checked="true"]::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71'%3E%3C/path%3E%3Cpath d='M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71'%3E%3C/path%3E%3C/svg%3E");
}

/* ══════════════════════════════════════════════════════════════════════════════
   MAIN SCREEN TOP HEADER & ONE-LINE SYSTEM STATUS
══════════════════════════════════════════════════════════════════════════════ */
.main-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 0 1.2rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid #e2e8f0;
}
.one-line-status {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  padding: 6px 16px;
  border-radius: 100px;
  font-size: 0.82rem;
  color: #334155;
  box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.status-pulse-dot {
  width: 8px; height: 8px;
  background: #22c55e;
  border-radius: 50%;
  box-shadow: 0 0 8px rgba(34,197,94,0.6);
  animation: pulseDot 2s ease infinite;
}
.status-divider {
  color: #cbd5e1;
  font-size: 0.75rem;
}
.brand-wordmark {
  font-size: 1.15rem;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.4px;
}
.brand-wordmark span { color: #f59e0b; }

/* ══════════════════════════════════════════════════════════════════════════════
   HERO / MAIN SECTION
══════════════════════════════════════════════════════════════════════════════ */
.hero-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 2.2rem 2.5rem;
  margin-bottom: 1.8rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.03);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 2rem;
  animation: fadeUp 0.45s ease both;
}
.hero-title {
  font-size: 2.5rem;
  font-weight: 800;
  line-height: 1.15;
  color: #0f172a;
  letter-spacing: -1px;
  margin-bottom: 0.6rem;
}
.hero-title span {
  color: #f59e0b;
}
.hero-desc {
  font-size: 0.95rem;
  color: #64748b;
  max-width: 520px;
  line-height: 1.6;
}
.hero-badge {
  display: inline-block;
  background: #fff7ed;
  border: 1px solid #ffedd5;
  color: #c2410c;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 100px;
  margin-bottom: 0.9rem;
}

/* ══════════════════════════════════════════════════════════════════════════════
   CARDS & KPI METRICS
══════════════════════════════════════════════════════════════════════════════ */
.card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 1.3rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.card:hover {
  box-shadow: 0 6px 20px rgba(0,0,0,0.06);
  transform: translateY(-2px);
}
.card-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: #94a3b8;
  margin-bottom: 0.9rem;
}

/* Hazard indicator widget */
.haz-score {
  font-size: 4rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -2px;
}
.haz-denom { font-size: 1.2rem; font-weight: 400; color: #94a3b8; }
.haz-rating-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin: 6px 0 12px;
}
.haz-track {
  background: #f1f5f9;
  border-radius: 100px;
  height: 6px;
  overflow: hidden;
  margin-bottom: 8px;
}
.haz-fill {
  height: 6px;
  border-radius: 100px;
  background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%);
}

.col-low       { color: #16a34a; }
.col-moderate  { color: #d97706; }
.col-high      { color: #ea580c; }
.col-very-high { color: #dc2626; }

/* KPI Grid */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.9rem;
  margin: 1.2rem 0;
}
.kpi-block {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
  transition: transform 0.18s ease;
}
.kpi-block:hover { transform: translateY(-2px); }
.kpi-num { font-size: 1.9rem; font-weight: 800; color: #0f172a; line-height: 1; letter-spacing: -0.5px; }
.kpi-lbl { font-size: 0.75rem; color: #64748b; margin-top: 4px; font-weight: 500; }

/* Stat rows */
.srow {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 0;
  border-bottom: 1px solid #f1f5f9;
  font-size: 0.875rem;
}
.srow:last-child { border-bottom: none; }
.srow-name { color: #64748b; font-weight: 500; }
.srow-val  { font-weight: 700; color: #0f172a; font-size: 0.9rem; }

/* Buttons */
div.stButton > button {
  background: #0f172a !important;
  color: #ffffff !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.55rem 1.6rem !important;
  transition: all 0.18s ease !important;
}
div.stButton > button:hover {
  background: #1e293b !important;
  box-shadow: 0 4px 14px rgba(15,23,42,0.2) !important;
  transform: translateY(-1px) !important;
}

div.stDownloadButton > button {
  background: #ffffff !important;
  color: #334155 !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  transition: all 0.18s ease !important;
}
div.stDownloadButton > button:hover {
  background: #f8fafc !important;
  border-color: #f59e0b !important;
}

/* File Uploader Dropzone */
div[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border: 1.5px dashed #cbd5e1 !important;
  border-radius: 14px !important;
  transition: border-color 0.2s !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: #f59e0b !important;
}
.dropzone {
  background: #ffffff;
  border: 1.5px dashed #cbd5e1;
  border-radius: 16px;
  padding: 3rem 2rem;
  text-align: center;
  transition: border-color 0.2s ease;
}
.dropzone:hover { border-color: #f59e0b; }
.dz-title { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem; }
.dz-sub   { font-size: 0.82rem; color: #94a3b8; }

/* Live Badge */
.live-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #dc2626;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  padding: 5px 12px;
  border-radius: 100px;
}

/* Status Pills */
.pill-pass { display:inline-block; background:#f0fdf4; border:1px solid #bbf7d0; color:#16a34a; font-size:.7rem; font-weight:700; padding:3px 10px; border-radius:100px; }
.pill-fail { display:inline-block; background:#fef2f2; border:1px solid #fecaca; color:#dc2626; font-size:.7rem; font-weight:700; padding:3px 10px; border-radius:100px; }
.pill-warn { display:inline-block; background:#fff7ed; border:1px solid #ffedd5; color:#ea580c; font-size:.7rem; font-weight:700; padding:3px 10px; border-radius:100px; }
.tag { display:inline-block; background:#f1f5f9; border:1px solid #e2e8f0; color:#475569; font-size:.7rem; font-weight:500; padding:3px 10px; border-radius:100px; margin:3px 2px; }

/* Source Cards */
.scard {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.1rem 1.4rem;
  margin-bottom: 0.65rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: all 0.2s ease;
}
.scard:hover { border-color: #cbd5e1; transform: translateY(-1px); }
.scard-name { font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 3px; }
.scard-meta { font-size: 0.76rem; color: #94a3b8; }
.scard-link {
  background: #0f172a;
  color: #ffffff !important;
  padding: 0.45rem 1rem;
  border-radius: 8px;
  text-decoration: none !important;
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
}

/* History items */
.hitem {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem 1.4rem;
  margin-bottom: 0.6rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.hitem-name { font-size: 0.9rem; font-weight: 600; color: #0f172a; margin-bottom: 3px; }
.hitem-meta { font-size: 0.75rem; color: #94a3b8; }
.hitem-score { font-size: 1.6rem; font-weight: 800; color: #f59e0b; text-align: right; line-height: 1; }
.hitem-rating { font-size: 0.68rem; font-weight: 700; text-align: right; margin-top: 3px; }

div[data-testid="stAlert"] {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
  color: #334155 !important;
}
div[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
}
div[data-testid="stSelectbox"] > div > div {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
}
.stSpinner > div { border-top-color: #f59e0b !important; }
video { border-radius: 12px !important; }
div[data-testid="stTabs"] [aria-selected="true"] {
  color: #0f172a !important;
  border-bottom: 2px solid #f59e0b !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─── Session State Initialization ─────────────────────────────────────────────
if "history"             not in st.session_state: st.session_state["history"] = []
if "reset_key"           not in st.session_state: st.session_state["reset_key"] = 0
if "video_result"        not in st.session_state: st.session_state["video_result"] = None
if "live_camera_running" not in st.session_state: st.session_state["live_camera_running"] = False
if "active_page"         not in st.session_state: st.session_state["active_page"] = "Overview"

# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initializing Neural Models...")
def load_models():
    return ImagePipeline(), VideoPipeline()

image_pipeline, video_pipeline = load_models()
statuses     = image_pipeline.get_component_statuses()
active_count = sum(1 for s in statuses.values() if s == "PASS")
total_count  = len(statuses)

# ─── SIDEBAR (Collapsible SVG Icon Bar) ───────────────────────────────────────
# Clean SVG Icons for Navigation (No emojis)
SVG_ICONS = {
    "Overview": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>',
    "Analyze Image": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>',
    "Analyze Video": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line></svg>',
    "Live Camera": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"></path><rect x="1" y="5" width="15" height="14" rx="2"></rect></svg>',
    "History": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
    "Model Status": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line></svg>',
    "About": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
    "Sources": '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>'
}

with st.sidebar:
    st.markdown("""
    <div class="sb-logo-box">
      <div class="sb-logo-mark">R</div>
      <div class="sb-logo-name">ROAD//IQ</div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = ["Overview", "Analyze Image", "Analyze Video", "Live Camera", "History", "Model Status", "About", "Sources"]
    
    selected_page = st.radio(
        "_sidebar_nav",
        nav_options,
        index=nav_options.index(st.session_state["active_page"]) if st.session_state["active_page"] in nav_options else 0,
        label_visibility="collapsed"
    )
    st.session_state["active_page"] = selected_page

PAGE = st.session_state["active_page"]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def reset_analysis():
    st.session_state["reset_key"] += 1
    st.session_state["video_result"] = None
    st.session_state["live_camera_running"] = False
    st.rerun()

def risk_col(r: str) -> str:
    return {
        "LOW": "col-low", "MODERATE": "col-moderate",
        "HIGH": "col-high", "VERY HIGH": "col-very-high",
    }.get(r.upper(), "col-moderate")

# ONE-LINE MAIN SCREEN STATUS HEADER (Top Left)
def render_main_header(active_mode: str):
    st.markdown(f"""
    <div class="main-header-bar">
      <div class="one-line-status">
        <span class="status-pulse-dot"></span>
        <strong>SYSTEM ONLINE</strong>
        <span class="status-divider">•</span>
        <span>{active_count}/{total_count} Models Active</span>
        <span class="status-divider">•</span>
        <span>Mode: {active_mode}</span>
      </div>
      <div class="brand-wordmark">ROAD<span>//IQ</span></div>
    </div>
    """, unsafe_allow_html=True)

def render_hero(badge: str, title_html: str, subtitle: str = "", show_reset: bool = True):
    h_col, btn_col = st.columns([5.5, 1])
    with h_col:
        st.markdown(f"""
        <div class="hero-card">
          <div>
            <div class="hero-badge">{badge}</div>
            <div class="hero-title">{title_html}</div>
            <div class="hero-desc">{subtitle}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with btn_col:
        if show_reset:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.button("＋ New Analysis", on_click=reset_analysis, use_container_width=True)

def hazard_card(score: int, rating: str):
    rc = risk_col(rating)
    st.markdown(f"""
    <div style="padding:.2rem 0 .8rem;text-align:center;">
      <div class="haz-score {rc}">{score}<span class="haz-denom"> /100</span></div>
      <div class="haz-rating-label {rc}">● {rating} Risk</div>
      <div class="haz-track">
        <div class="haz-fill" style="width:{min(score,100)}%"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def det_rows(counts: dict):
    rows = [
        ("Cars",         counts.get("car", 0)),
        ("Motorcycles",  counts.get("motorcycle", 0)),
        ("Pedestrians",  counts.get("person", 0) + counts.get("pedestrian", 0)),
        ("Bus / Truck",   counts.get("bus", 0) + counts.get("truck", 0)),
    ]
    st.markdown(
        "".join(f'<div class="srow"><span class="srow-name">{n}</span>'
                f'<span class="srow-val">{v:02d}</span></div>' for n, v in rows),
        unsafe_allow_html=True,
    )

def kpi_row(objs, dmgs, signs, drv):
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-block">
        <div class="kpi-num">{objs}</div>
        <div class="kpi-lbl">Road Users</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-num">{dmgs}</div>
        <div class="kpi-lbl">Road Damage</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-num">{signs}</div>
        <div class="kpi-lbl">Traffic Signs</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-num">{drv}</div>
        <div class="kpi-lbl">Drivable Area</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 1: OVERVIEW / ANALYZE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
if PAGE in ["Overview", "Analyze Image"]:
    render_main_header("Image Pipeline")
    render_hero(
        badge="Multi-Model Computer Vision",
        title_html="Indian Road <span>Intelligence</span>",
        subtitle="Upload a road image to run real-time inference — detecting vehicles, "
                 "road damage, potholes, drivable road surface, and computing a hazard score.",
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key=f"img_{st.session_state['reset_key']}",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="dropzone">
          <div class="dz-title">Drop a road image here</div>
          <div class="dz-sub">Supports JPG, PNG · up to 200 MB</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 Select **Live Camera** in the left navigation for real-time webcam feed analysis.")
    else:
        file_bytes  = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr   = cv2.imdecode(file_bytes, 1)
        image_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Executing neural inference..."):
            annotated_bgr, json_report = image_pipeline.process(
                image_bgr, source_name=uploaded_file.name
            )
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        summary       = json_report.get("summary", {})
        hazard_score  = summary.get("road_hazard_indicator", 0)
        hazard_rating = summary.get("hazard_rating", "LOW")

        h_item = {
            "file_name":    uploaded_file.name,
            "time":         time.strftime("%H:%M:%S"),
            "objects":      summary.get("total_road_users_detected", 0),
            "damage":       summary.get("total_damage_anomalies_detected", 0),
            "hazard_score": hazard_score,
            "rating":       hazard_rating,
        }
        if h_item not in st.session_state["history"]:
            st.session_state["history"].append(h_item)

        img_col, side_col = st.columns([1.95, 1], gap="large")

        with img_col:
            st.markdown('<div class="card-label">Annotated Analysis Output</div>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)

            obj_cnt   = summary.get("total_road_users_detected", 0)
            dmg_cnt   = summary.get("total_damage_anomalies_detected", 0)
            signs_cnt = summary.get("total_traffic_signs_detected", 0)
            signs_str = str(signs_cnt) if statuses.get("traffic_sign_detector") == "PASS" else "—"
            drivable  = json_report.get("road_segmentation", {}).get("drivable_area_percentage")
            drv_str   = f"{drivable}%" if drivable is not None else "—"
            kpi_row(obj_cnt, dmg_cnt, signs_str, drv_str)

            json_bytes = json.dumps(json_report, indent=2).encode("utf-8")
            st.download_button(
                "Export Technical Report (.json)",
                data=json_bytes,
                file_name=f"report_{uploaded_file.name}.json",
                mime="application/json",
                use_container_width=True,
            )

        with side_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Road Hazard Indicator</div>', unsafe_allow_html=True)
            hazard_card(hazard_score, hazard_rating)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Detection Breakdown</div>', unsafe_allow_html=True)
            counts = json_report.get("counts_breakdown", {}).get("road_users", {})
            det_rows(counts)
            st.markdown('</div>', unsafe_allow_html=True)

            prox = json_report.get("spatial_proximity_alerts", [])
            if prox:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">Proximity Alerts</div>', unsafe_allow_html=True)
                for a in prox[:4]:
                    lvl = a.get("risk_proximity", "")
                    clr = "#dc2626" if lvl == "CRITICAL" else "#ea580c"
                    st.markdown(
                        f'<div class="srow">'
                        f'<span class="srow-name" style="color:{clr}">Alert: {lvl}</span>'
                        f'<span class="srow-val">{a.get("distance_px",0):.0f} px</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("View Full JSON Report"):
                st.json(json_report)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 2: ANALYZE VIDEO
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Analyze Video":
    render_main_header("Video Pipeline")
    render_hero(
        badge="Video Scene Processing",
        title_html="Road Video <span>Pipeline</span>",
        subtitle="Upload a road video. The model architecture processes every frame and encodes a web-compatible H.264 annotated video.",
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key=f"vid_{st.session_state['reset_key']}",
    )

    if uploaded_video is None:
        st.markdown("""
        <div class="dropzone">
          <div class="dz-title">Drop a road video here</div>
          <div class="dz-sub">Supports MP4, AVI, MOV, MKV</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        tab_src, tab_out = st.tabs(["Source Video", "Annotated Video Output"])

        with tab_src:
            st.markdown("<br>", unsafe_allow_html=True)
            st.video(tfile.name)
            proc_col, _ = st.columns([1.2, 2])
            with proc_col:
                if st.button("Process Video Frames", use_container_width=True):
                    with st.spinner("Processing video frames & encoding H.264 video..."):
                        out_path = os.path.join(
                            tempfile.gettempdir(), f"analyzed_{uploaded_video.name}.mp4"
                        )
                        res_path, video_report = video_pipeline.process_video(
                            video_input_path=tfile.name,
                            output_video_path=out_path,
                            frame_stride=2,
                        )
                        with open(res_path, "rb") as vf:
                            video_bytes = vf.read()
                        st.session_state["video_result"] = {
                            "video_bytes": video_bytes,
                            "file_name":   f"analyzed_{uploaded_video.name}.mp4",
                            "report":      video_report,
                        }
                    st.success("✅ Video analysis complete!")

        with tab_out:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.session_state["video_result"] is not None:
                vr  = st.session_state["video_result"]
                vb  = vr["video_bytes"]
                rep = vr["report"]

                st.video(vb)

                vhaz   = rep.get("video_hazard_summary", {})
                avg_h  = vhaz.get("average_road_hazard_indicator", 0)
                peak_h = vhaz.get("peak_road_hazard_indicator", 0)
                rating = vhaz.get("overall_hazard_rating", "N/A")
                frms   = rep.get("processed_frames_count", 0)
                rc     = risk_col(rating)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-grid">
                  <div class="kpi-block">
                    <div class="kpi-num {rc}">{avg_h}</div>
                    <div class="kpi-lbl">Avg Hazard Score</div>
                  </div>
                  <div class="kpi-block">
                    <div class="kpi-num">{peak_h}</div>
                    <div class="kpi-lbl">Peak Hazard Score</div>
                  </div>
                  <div class="kpi-block">
                    <div class="kpi-num {rc}" style="font-size:1.4rem">{rating}</div>
                    <div class="kpi-lbl">Overall Rating</div>
                  </div>
                  <div class="kpi-block">
                    <div class="kpi-num">{frms}</div>
                    <div class="kpi-lbl">Frames Processed</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                dl1, dl2 = st.columns(2, gap="medium")
                dl1.download_button(
                    "Download Output Video (.mp4)",
                    data=vb,
                    file_name=vr["file_name"],
                    mime="video/mp4",
                    use_container_width=True,
                )
                dl2.download_button(
                    "Download Video Report (.json)",
                    data=json.dumps(rep, indent=2),
                    file_name=f"report_{uploaded_video.name}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("Click **Process Video Frames** in the Source Video tab to analyze.")

        try: os.remove(tfile.name)
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 3: LIVE CAMERA
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Live Camera":
    render_main_header("Real-Time Webcam Feed")
    render_hero(
        badge="Real-Time Streaming",
        title_html="Continuous Live <span>Feed</span>",
        subtitle="Streams your camera continuously, running instant per-frame neural detection, hazard evaluation, and count summary.",
        show_reset=False,
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 3], gap="small")
    with ctrl1:
        if not st.session_state["live_camera_running"]:
            if st.button("Start Live Stream", use_container_width=True):
                st.session_state["live_camera_running"] = True
                st.rerun()
        else:
            if st.button("Stop Stream", use_container_width=True):
                st.session_state["live_camera_running"] = False
                st.rerun()
    with ctrl2:
        cam_src = st.selectbox("Device Index", [0, 1, 2], index=0, label_visibility="collapsed")
    with ctrl3:
        if st.session_state["live_camera_running"]:
            st.markdown(
                '<div style="padding-top:6px">'
                '<span class="live-badge"><span class="status-pulse-dot"></span>LIVE FEED ACTIVE</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:#e2e8f0;margin:1.5rem 0'>", unsafe_allow_html=True)

    if st.session_state["live_camera_running"]:
        cap = cv2.VideoCapture(cam_src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error(f"Cannot open camera device at index {cam_src}.")
            st.session_state["live_camera_running"] = False
        else:
            feed_col, hud_col = st.columns([1.95, 1], gap="large")
            with feed_col:
                st.markdown('<div class="card-label">Live Camera Feed</div>', unsafe_allow_html=True)
                frame_ph = st.empty()
            with hud_col:
                st.markdown('<div class="card-label">Real-Time Hazard HUD</div>', unsafe_allow_html=True)
                hud_ph = st.empty()

            try:
                while st.session_state["live_camera_running"]:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Camera stream disconnected.")
                        break

                    ann_bgr, live_rep = image_pipeline.process(frame, source_name="live_camera_stream")
                    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    frame_ph.image(ann_rgb, use_container_width=True)

                    ls    = live_rep.get("summary", {})
                    score = ls.get("road_hazard_indicator", 0)
                    rate  = ls.get("hazard_rating", "LOW")
                    rc    = risk_col(rate)
                    c     = live_rep.get("counts_breakdown", {}).get("road_users", {})

                    hud_ph.markdown(f"""
                    <div class="card" style="text-align:center;margin-bottom:.8rem">
                      <div class="haz-score {rc}">{score}<span class="haz-denom"> /100</span></div>
                      <div class="haz-rating-label {rc}">● {rate} Risk</div>
                      <div class="haz-track"><div class="haz-fill" style="width:{min(score,100)}%"></div></div>
                    </div>
                    <div class="card">
                      <div class="srow"><span class="srow-name">Cars</span><span class="srow-val">{c.get('car',0):02d}</span></div>
                      <div class="srow"><span class="srow-name">Motorcycles</span><span class="srow-val">{c.get('motorcycle',0):02d}</span></div>
                      <div class="srow"><span class="srow-name">People</span><span class="srow-val">{c.get('person',0)+c.get('pedestrian',0):02d}</span></div>
                      <div class="srow" style="border:none"><span class="srow-name">Bus/Truck</span><span class="srow-val">{c.get('bus',0)+c.get('truck',0):02d}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    time.sleep(0.03)
            finally:
                cap.release()
    else:
        st.markdown("""
        <div class="dropzone">
          <div class="dz-title">Camera Feed Idle</div>
          <div class="dz-sub">Click <strong>Start Live Stream</strong> above to connect camera.</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 4: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "History":
    render_main_header("Session Logs")
    render_hero(
        badge="Session Records",
        title_html="Analysis <span>History</span>",
        subtitle="View previous image and video analysis reports generated during this session.",
        show_reset=False,
    )
    if not st.session_state["history"]:
        st.markdown("""
        <div class="dropzone">
          <div class="dz-title">No Recorded History</div>
          <div class="dz-sub">Upload an image or video to log session analysis reports.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state["history"]):
            rc = risk_col(item["rating"])
            st.markdown(f"""
            <div class="hitem">
              <div>
                <div class="hitem-name">{item['file_name']}</div>
                <div class="hitem-meta">
                  Time: {item['time']} &nbsp;•&nbsp;
                  {item['objects']} objects detected &nbsp;•&nbsp;
                  {item['damage']} damage anomalies
                </div>
              </div>
              <div>
                <div class="hitem-score">{item['hazard_score']}<span style="font-size:.85rem;color:#94a3b8"> /100</span></div>
                <div class="hitem-rating {rc}">{item['rating']} RISK</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 5: MODEL STATUS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Model Status":
    render_main_header("Component Verification")
    render_hero(
        badge="Architecture Architecture",
        title_html="Neural Model <span>Status</span>",
        subtitle="Verification matrix of all four computer vision components loaded into the pipeline.",
        show_reset=False,
    )
    MDESC = {
        "generic_object_detector": ("YOLOv8n Object Detector (COCO)", "Detects 7 COCO road user categories: car, motorcycle, bus, truck, bicycle, person, traffic light."),
        "road_damage_detector":    ("RoadScanCV Damage Detector", "Detects potholes, cracks, and manholes. Evaluates pothole severity via iWatchRoad heuristic."),
        "road_segmenter":          ("IDD ResNet50 U-Net Segmenter", "Pixel-level drivable road surface segmentation trained on IIIT Hyderabad IDD dataset."),
        "traffic_sign_detector":   ("Indian Traffic Sign Detector", "56-class Indian traffic sign detector architecture."),
    }
    c1, c2 = st.columns(2, gap="large")
    for i, (k, v) in enumerate(statuses.items()):
        name, desc = MDESC.get(k, (k, ""))
        badge = f'<span class="pill-pass">● ACTIVE (PASS)</span>'
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div style="font-size:.98rem;font-weight:700;color:#0f172a">{name}</div>
                {badge}
              </div>
              <div style="font-size:.82rem;color:#64748b;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 6: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "About":
    render_main_header("Project Overview")
    render_hero(
        badge="System Architecture",
        title_html="Indian Road <span>Intelligence System</span>",
        subtitle="An integrated computer vision framework purpose-built for Indian road scenes and autonomous driving research.",
        show_reset=False,
    )
    FEATS = [
        ("Multi-Class Road User Detection", "YOLOv8n detects 7 COCO classes: cars, motorcycles, auto-rickshaws, buses, trucks, bicycles, pedestrians."),
        ("Road Damage & Severity Analysis", "RoadScanCV YOLOv8 identifies potholes, cracks, manholes. Severity graded via iWatchRoad pixel area heuristic."),
        ("Drivable Area Segmentation", "ResNet50 U-Net on IIIT Hyderabad IDD dataset produces pixel-level drivable road surface coverage masks."),
        ("Spatial Proximity Engine", "Euclidean distance matrix flags user–hazard proximity alerts at 150 px (Warning) and 75 px (Critical)."),
        ("Road Hazard Indicator (0–100)", "Transparent engineering formula: 40% user density + 35% damage + 15% drivable loss + 10% proximity."),
        ("Real-Time Webcam Feed", "Continuous webcam streaming with instant per-frame inference and hazard metrics."),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i, (title, desc) in enumerate(FEATS):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="card">
              <div style="font-size:.95rem;font-weight:700;color:#0f172a;margin-bottom:6px">{title}</div>
              <div style="font-size:.82rem;color:#64748b;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 7: SOURCES
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Sources":
    render_main_header("Attribution & Open Source")
    render_hero(
        badge="Repository Attribution",
        title_html="Open-Source <span>Sources</span>",
        subtitle="All five open-source repositories integrated into the ROAD//IQ system.",
        show_reset=False,
    )
    SRCS = [
        ("IDD-Indian-Driving-Dataset", "Multi-Label Scene Classification · Indian road scenes", "IIIT Hyderabad / AutoNUE", "Apache 2.0", "https://github.com/idd-dataset/IDD-Indian-Driving-Dataset"),
        ("idd-road-segmentation", "Drivable Surface Segmentation · ResNet50 U-Net", "IDD Community", "MIT", "https://github.com/idd-dataset/idd-road-segmentation"),
        ("Indian-Traffic-Sign-Detection-System", "Indian Traffic Signboard Detection · YOLOv8s · 56 classes", "Akanksha", "Unlicensed", "https://github.com/akanksha-2002/Indian_traffic_sign_detection_system"),
        ("iWatchRoad", "Pothole Tracking & Severity Grading · pixel-area heuristic", "SMLab NISER", "CC BY-NC-SA 4.0", "https://github.com/smlab-niser/iWatchRoad"),
        ("RoadScanCV", "Road Damage Detection · Potholes, Cracks, Manholes", "Botkraker", "Unlicensed", "https://github.com/Botkraker/RoadScanCV"),
    ]
    for name, task, author, lic, url in SRCS:
        st.markdown(f"""
        <div class="scard">
          <div>
            <div class="scard-name">{name}</div>
            <div class="scard-meta">{task}</div>
            <div class="scard-meta">Author: {author} &nbsp;•&nbsp; License: {lic}</div>
          </div>
          <a href="{url}" target="_blank" class="scard-link">GitHub ↗</a>
        </div>
        """, unsafe_allow_html=True)
