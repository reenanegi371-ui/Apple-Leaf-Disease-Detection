"""
🍎 Apple Leaf Disease Detection System
Full Streamlit app using YOLOv8 (best.pt) with live camera + image upload
Detects: Apple Scab, Black Rot, Cedar Apple Rust, Healthy leaves
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime
import io
import base64

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="LeafScan · Apple Disease AI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# DISEASE DATABASE
# ══════════════════════════════════════════════
DISEASE_INFO = {
    "apple_scab": {
        "display": "Apple Scab",
        "severity": "Moderate",
        "color": "#8B6914",
        "bg": "rgba(139,105,20,0.12)",
        "icon": "🟤",
        "description": "Caused by the fungus Venturia inaequalis. Appears as olive-green to brown velvety spots on leaves.",
        "symptoms": ["Olive-green or brown velvety lesions", "Yellowing around infected areas", "Premature leaf drop", "Distorted leaves"],
        "treatment": ["Apply fungicide (Captan, Mancozeb) at bud break", "Remove and destroy infected leaves", "Prune for better air circulation", "Apply dormant oil spray in early spring"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Rake and destroy fallen leaves", "Apply lime sulfur before bud break"],
        "severity_score": 6,
    },
    "black_rot": {
        "display": "Black Rot",
        "severity": "Severe",
        "color": "#c0392b",
        "bg": "rgba(192,57,43,0.12)",
        "icon": "🔴",
        "description": "Caused by Botryosphaeria obtusa. Produces circular lesions with purple margins that turn brown-black.",
        "symptoms": ["Circular lesions with purple margins", "Brown-black center with concentric rings", "Frog-eye appearance", "Cankers on branches"],
        "treatment": ["Remove and destroy infected plant parts", "Apply copper-based fungicide", "Prune cankers 15cm beyond visible infection", "Bordeaux mixture applications"],
        "prevention": ["Remove mummified fruits and dead wood", "Maintain tree vigor through fertilization", "Avoid wounding bark", "Proper spacing for air circulation"],
        "severity_score": 9,
    },
    "cedar_apple_rust": {
        "display": "Cedar Apple Rust",
        "severity": "High",
        "color": "#e67e22",
        "bg": "rgba(230,126,34,0.12)",
        "icon": "🟠",
        "description": "Caused by Gymnosporangium juniperi-virginianae. Requires both cedar/juniper and apple as alternate hosts.",
        "symptoms": ["Bright orange-yellow spots on upper leaf surface", "Tube-like structures on leaf undersides", "Premature defoliation", "Fruit deformation"],
        "treatment": ["Apply myclobutanil or triadimefon fungicide", "Start treatments at pink bud stage", "Repeat every 7–10 days during wet spring", "Remove nearby juniper/cedar if possible"],
        "prevention": ["Plant resistant apple varieties", "Remove nearby juniper galls in winter", "Avoid planting apple near cedar trees", "Apply protective fungicides in spring"],
        "severity_score": 7,
    },
    "healthy": {
        "display": "Healthy Leaf",
        "severity": "None",
        "color": "#27ae60",
        "bg": "rgba(39,174,96,0.12)",
        "icon": "🟢",
        "description": "The leaf shows no signs of disease. Continue regular monitoring and preventive care.",
        "symptoms": ["No visible lesions", "Uniform green color", "Normal leaf structure", "Healthy veination"],
        "treatment": ["No treatment required", "Maintain regular watering schedule", "Continue balanced fertilization", "Monitor periodically"],
        "prevention": ["Regular scouting every 7–10 days", "Maintain tree health with proper nutrition", "Ensure good air circulation", "Remove fallen leaves in autumn"],
        "severity_score": 0,
    },
    "unknown": {
        "display": "Unknown Class",
        "severity": "—",
        "color": "#7f8c8d",
        "bg": "rgba(127,140,141,0.12)",
        "icon": "⚪",
        "description": "The model has detected an object with a custom class label from your training data.",
        "symptoms": ["Refer to your dataset labels"],
        "treatment": ["Refer to domain-specific guidance"],
        "prevention": ["Monitor regularly"],
        "severity_score": 5,
    },
}

def get_disease_info(class_name: str) -> dict:
    """Match detected class name to disease info."""
    cn = class_name.lower().replace(" ", "_").replace("-", "_")
    for key in DISEASE_INFO:
        if key in cn or cn in key:
            return DISEASE_INFO[key]
    # Try partial match
    for key, val in DISEASE_INFO.items():
        if key.split("_")[0] in cn:
            return val
    info = DISEASE_INFO["unknown"].copy()
    info["display"] = class_name
    return info


# ══════════════════════════════════════════════
# CSS  –  Botanical Luxury Theme
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:          #f5f2ec;
    --bg2:         #ede9e0;
    --surface:     #ffffff;
    --border:      #d8d0c4;
    --border2:     #c8bfb0;
    --forest:      #1a3a2a;
    --leaf:        #2d6a4f;
    --leaf-light:  #52b788;
    --bark:        #6b4c2a;
    --cream:       #faf7f2;
    --text:        #1e2d1e;
    --text2:       #4a5c4a;
    --muted:       #8a9a8a;
    --gold:        #c9a84c;
    --shadow:      rgba(30,45,30,0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--forest) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #c8d8c8 !important; }
[data-testid="stSidebar"] label { color: #8aaa8a !important; font-size: 0.72rem !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div input {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #e8f0e8 !important;
}

/* ── Hero ──────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, var(--forest) 0%, #2d5a3d 60%, #1a3a2a 100%);
    border-radius: 12px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(26,58,42,0.25);
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-leaf-bg {
    position: absolute;
    right: -20px; top: -30px;
    font-size: 12rem;
    opacity: 0.04;
    pointer-events: none;
    line-height: 1;
    transform: rotate(-15deg);
}
.hero-tag {
    display: inline-block;
    background: rgba(82,183,136,0.2);
    border: 1px solid rgba(82,183,136,0.4);
    color: var(--leaf-light) !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #f0ede6 !important;
    line-height: 1.05;
    margin: 0 0 10px;
    letter-spacing: -0.5px;
}
.hero-title em {
    font-style: italic;
    color: var(--leaf-light) !important;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: rgba(200,216,200,0.75) !important;
    max-width: 480px;
    line-height: 1.6;
}
.hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 24px;
}
.hero-stat-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--leaf-light) !important;
    line-height: 1;
}
.hero-stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: rgba(180,200,180,0.6) !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
}

/* ── Tabs / Mode selector ────────────────── */
.mode-tabs {
    display: flex;
    gap: 0;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
    margin-bottom: 24px;
    width: fit-content;
}
.mode-tab {
    padding: 8px 22px;
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
    background: transparent;
    color: var(--muted);
}
.mode-tab.active {
    background: var(--forest);
    color: #f0ede6 !important;
    box-shadow: 0 2px 8px rgba(26,58,42,0.2);
}

/* ── Result card ─────────────────────────── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 20px var(--shadow);
}
.result-card-header {
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
}
.result-card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text);
}
.result-card-body { padding: 20px 24px; }

/* ── Disease badge ───────────────────────── */
.disease-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── Severity bar ────────────────────────── */
.sev-bar-wrap {
    background: var(--bg2);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin: 6px 0 14px;
}
.sev-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* ── Info sections ───────────────────────── */
.info-section {
    margin-bottom: 20px;
}
.info-section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border);
}
.info-item {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    font-size: 0.84rem;
    color: var(--text2);
    margin-bottom: 5px;
    line-height: 1.5;
}
.info-item::before {
    content: '›';
    color: var(--leaf);
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Confidence meter ────────────────────── */
.conf-meter {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 14px;
}
.conf-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 2px;
}
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
}

/* ── Detection list (multi) ──────────────── */
.det-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 6px;
    border: 1px solid var(--border);
    background: var(--cream);
    transition: box-shadow 0.2s;
}
.det-row:hover { box-shadow: 0 2px 12px var(--shadow); }
.det-name {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
    color: var(--text);
}
.det-conf-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 500;
}

/* ── Metric chips ────────────────────────── */
.chip-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 20px;
}
.chip {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    text-align: center;
}
.chip-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--forest);
    line-height: 1;
}
.chip-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-top: 3px;
}

/* ── Live status ─────────────────────────── */
.live-dot {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-live    { background: #27ae60; box-shadow: 0 0 6px #27ae60; animation: pulse 1.2s infinite; }
.dot-idle    { background: var(--muted); }
.dot-error   { background: #e74c3c; }
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.3; }
}

/* ── History log ─────────────────────────── */
.hist-item {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    background: var(--cream);
    border: 1px solid var(--border);
    font-size: 0.8rem;
}
.hist-time {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    min-width: 58px;
}

/* ── Buttons ─────────────────────────────── */
.stButton > button {
    background: var(--forest) !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 22px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: var(--leaf) !important;
    box-shadow: 0 4px 20px rgba(45,106,79,0.35) !important;
    transform: translateY(-1px) !important;
}
button[kind="secondary"] {
    background: transparent !important;
    color: var(--text2) !important;
    border: 1px solid var(--border) !important;
}
button[kind="secondary"]:hover {
    background: var(--bg2) !important;
    transform: none !important;
}

/* ── Sliders ─────────────────────────────── */
[data-testid="stSlider"] > div > div { accent-color: var(--leaf) !important; }

/* ── File uploader ───────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background: var(--cream) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 10px !important;
}

/* ── Divider ─────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Hide chrome ─────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* sidebar labels */
.sidebar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(140,170,140,0.7);
    margin: 18px 0 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════
ss = st.session_state
for k, v in {
    "mode": "upload",
    "cam_running": False,
    "frame_count": 0,
    "fps": 0.0,
    "history": [],        # list of {time, disease, conf}
    "last_dets": [],
    "total_frames": 0,
}.items():
    if k not in ss:
        ss[k] = v


# ══════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_yolo(path: str):
    try:
        from ultralytics import YOLO
        m = YOLO(path)
        return m, None
    except ImportError:
        return None, "ultralytics not installed → pip install ultralytics"
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 16px 8px;'>
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.6rem;
                    font-weight:700;color:#c8d8c8;letter-spacing:-0.3px;'>
            🍃 LeafScan
        </div>
        <div style='font-family:"DM Mono",monospace;font-size:0.58rem;
                    color:rgba(140,170,140,0.6);letter-spacing:2px;
                    text-transform:uppercase;margin-top:3px;'>
            Apple Disease AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Model Weights</div>', unsafe_allow_html=True)
    model_path = st.text_input(
        "model_path", value="best.pt",
        label_visibility="collapsed",
        placeholder="best.pt",
        help="Path to YOLOv8 .pt weights",
    )
    uploaded_w = st.file_uploader("Upload .pt", type=["pt"], label_visibility="collapsed")
    if uploaded_w:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp.write(uploaded_w.read())
        tmp.close()
        model_path = tmp.name

    st.markdown('<div class="sidebar-label">Detection Settings</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.01, format="%.2f")
    iou_thresh  = st.slider("IoU (NMS)", 0.10, 0.90, 0.50, 0.01, format="%.2f")
    img_size    = st.select_slider("Image size", [320, 416, 512, 640, 768, 1024], value=640)

    st.markdown('<div class="sidebar-label">Camera</div>', unsafe_allow_html=True)
    cam_idx  = st.selectbox("Camera index", [0, 1, 2, 3], label_visibility="collapsed")
    max_fps  = st.slider("Target FPS", 5, 30, 15)

    st.markdown('<div class="sidebar-label">Display</div>', unsafe_allow_html=True)
    show_conf   = st.checkbox("Show confidence", True)
    show_labels = st.checkbox("Show labels", True)
    box_color_name = st.selectbox(
        "Box color", ["Forest Green", "Gold", "White", "Red"],
        label_visibility="collapsed"
    )
    box_colors = {"Forest Green": (45,106,79), "Gold": (80,168,201), "White": (240,237,230), "Red": (46,64,210)}
    BOX_COLOR = box_colors[box_color_name]

    st.markdown("---")
    # Disease legend
    st.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.58rem;text-transform:uppercase;
                letter-spacing:2px;color:rgba(140,170,140,0.6);margin-bottom:10px;'>
        Disease Classes
    </div>""", unsafe_allow_html=True)
    for key, info in list(DISEASE_INFO.items())[:-1]:  # exclude "unknown"
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;padding:5px 0;
                    border-bottom:1px solid rgba(255,255,255,0.05);'>
            <span>{info["icon"]}</span>
            <span style='font-size:0.75rem;color:#c8d8c8;'>{info["display"]}</span>
            <span style='margin-left:auto;font-family:"DM Mono",monospace;font-size:0.6rem;
                         color:rgba(140,170,140,0.5);'>
                sev {info["severity_score"]}/10
            </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-leaf-bg">🍃</div>
    <div class="hero-tag">🍎 Apple Orchard Intelligence</div>
    <h1 class="hero-title">Apple Leaf<br><em>Disease Detection</em></h1>
    <p class="hero-sub">
        AI-powered plant pathology using YOLOv8. Detect Scab, Black Rot, 
        Cedar Apple Rust and more — in real-time or from images.
    </p>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-val">4</div>
            <div class="hero-stat-label">Disease Classes</div>
        </div>
        <div>
            <div class="hero-stat-val">YOLOv8</div>
            <div class="hero-stat-label">Model Backbone</div>
        </div>
        <div>
            <div class="hero-stat-val">Real-time</div>
            <div class="hero-stat-label">Inference Mode</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODE SELECTION
# ══════════════════════════════════════════════
tab_upload, tab_camera, tab_history = st.tabs(["📷  Image Upload", "🎥  Live Camera", "📋  History"])


# ══════════════════════════════════════════════
# HELPER: RENDER DETECTION RESULTS
# ══════════════════════════════════════════════
def render_single_result(name: str, conf: float):
    info = get_disease_info(name)
    sev_pct = info["severity_score"] * 10

    # Color logic for confidence
    if conf >= 0.75:
        conf_color = "#27ae60"
    elif conf >= 0.50:
        conf_color = "#f39c12"
    else:
        conf_color = "#e74c3c"

    st.markdown(f"""
    <div class="result-card" style="border-left:4px solid {info['color']};">
        <div class="result-card-header" style="background:{info['bg']};">
            <span style="font-size:1.8rem;">{info['icon']}</span>
            <div>
                <div class="result-card-title">{info['display']}</div>
                <span class="disease-badge" style="background:{info['bg']};
                      color:{info['color']};border:1px solid {info['color']}40;">
                    Severity: {info['severity']}
                </span>
            </div>
            <div class="conf-meter" style="margin-left:auto;min-width:120px;">
                <div class="conf-val" style="color:{conf_color};">{conf*100:.1f}%</div>
                <div class="conf-label">Confidence</div>
            </div>
        </div>
        <div class="result-card-body">
            <p style="color:var(--text2);font-size:0.88rem;line-height:1.7;margin-bottom:16px;">
                {info['description']}
            </p>
            <div style="margin-bottom:6px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                            text-transform:uppercase;letter-spacing:2px;color:var(--muted);
                            margin-bottom:4px;">Severity Score</div>
                <div class="sev-bar-wrap">
                    <div class="sev-bar-fill"
                         style="width:{sev_pct}%;background:{info['color']};"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 columns: symptoms / treatment / prevention
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="info-section">', unsafe_allow_html=True)
        st.markdown('<div class="info-section-title">🔍 Symptoms</div>', unsafe_allow_html=True)
        for s in info["symptoms"]:
            st.markdown(f'<div class="info-item">{s}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-section">', unsafe_allow_html=True)
        st.markdown('<div class="info-section-title">💊 Treatment</div>', unsafe_allow_html=True)
        for t in info["treatment"]:
            st.markdown(f'<div class="info-item">{t}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-section">', unsafe_allow_html=True)
        st.markdown('<div class="info-section-title">🛡 Prevention</div>', unsafe_allow_html=True)
        for p in info["prevention"]:
            st.markdown(f'<div class="info-item">{p}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def annotate_image(image_bgr, results, model, show_lbl=True, show_cf=True, bcolor=(45,106,79), thick=2):
    """Draw YOLO bounding boxes on image."""
    out = image_bgr.copy()
    r = results[0]
    if r.boxes is None:
        return out, []
    dets = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = model.names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        info   = get_disease_info(name)

        # Box
        cv2.rectangle(out, (x1,y1), (x2,y2), bcolor, thick)

        # Corner accents
        L = 15
        cv2.line(out, (x1,y1), (x1+L, y1), bcolor, thick+1)
        cv2.line(out, (x1,y1), (x1, y1+L), bcolor, thick+1)
        cv2.line(out, (x2,y1), (x2-L, y1), bcolor, thick+1)
        cv2.line(out, (x2,y1), (x2, y1+L), bcolor, thick+1)
        cv2.line(out, (x1,y2), (x1+L, y2), bcolor, thick+1)
        cv2.line(out, (x1,y2), (x1, y2-L), bcolor, thick+1)
        cv2.line(out, (x2,y2), (x2-L, y2), bcolor, thick+1)
        cv2.line(out, (x2,y2), (x2, y2-L), bcolor, thick+1)

        # Label
        if show_lbl or show_cf:
            label = ""
            if show_lbl: label += info["display"]
            if show_cf:  label += f"  {conf:.2f}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.rectangle(out, (x1, y1-lh-10), (x1+lw+10, y1), bcolor, -1)
            cv2.putText(out, label, (x1+5, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (240,237,230), 1, cv2.LINE_AA)

        dets.append({"name": name, "conf": conf, "display": info["display"],
                     "icon": info["icon"], "color": info["color"],
                     "bg": info["bg"], "severity": info["severity"]})
    return out, dets


# ══════════════════════════════════════════════
# TAB 1 — IMAGE UPLOAD
# ══════════════════════════════════════════════
with tab_upload:
    st.markdown("<br>", unsafe_allow_html=True)

    up_col, res_col = st.columns([1, 1.2], gap="large")

    with up_col:
        st.markdown("""
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.5rem;
                    font-weight:600;color:var(--forest);margin-bottom:14px;'>
            Upload Leaf Image
        </div>""", unsafe_allow_html=True)

        uploaded_img = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg","jpeg","png","bmp","webp","tiff"],
            label_visibility="visible",
        )

        if uploaded_img:
            img_pil   = Image.open(uploaded_img).convert("RGB")
            img_np    = np.array(img_pil)
            img_bgr   = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            st.image(img_pil, caption="Original Image", use_container_width=True)

            if st.button("🔬  Analyse Leaf", use_container_width=True):
                if not Path(model_path).exists():
                    st.error(f"Model not found: `{model_path}`")
                else:
                    with st.spinner("Running inference…"):
                        model, err = load_yolo(model_path)
                        if err:
                            st.error(f"Model error: {err}")
                        else:
                            t0 = time.time()
                            results = model.predict(
                                img_bgr,
                                conf=conf_thresh,
                                iou=iou_thresh,
                                imgsz=img_size,
                                verbose=False,
                            )
                            elapsed = (time.time() - t0) * 1000

                            annotated_bgr, dets = annotate_image(
                                img_bgr, results, model,
                                show_labels, show_conf, BOX_COLOR
                            )
                            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                            ss["last_dets"] = dets

                            # Save to history
                            for d in dets:
                                ss["history"].insert(0, {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "disease": d["display"],
                                    "conf": d["conf"],
                                    "icon": d["icon"],
                                    "source": "upload",
                                })
                            ss["history"] = ss["history"][:100]

                    with res_col:
                        st.markdown("""
                        <div style='font-family:"Cormorant Garamond",serif;font-size:1.5rem;
                                    font-weight:600;color:var(--forest);margin-bottom:14px;'>
                            Analysis Results
                        </div>""", unsafe_allow_html=True)

                        # Metrics chips
                        st.markdown(f"""
                        <div class="chip-row">
                            <div class="chip">
                                <div class="chip-val">{len(dets)}</div>
                                <div class="chip-lbl">Detections</div>
                            </div>
                            <div class="chip">
                                <div class="chip-val">{elapsed:.0f}ms</div>
                                <div class="chip-lbl">Inference Time</div>
                            </div>
                            <div class="chip">
                                <div class="chip-val">{max((d["conf"] for d in dets), default=0)*100:.0f}%</div>
                                <div class="chip-lbl">Top Confidence</div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                        # Annotated image
                        st.image(annotated_rgb, caption="Detected Regions", use_container_width=True)

                        # Download button
                        buf = io.BytesIO()
                        Image.fromarray(annotated_rgb).save(buf, format="PNG")
                        st.download_button(
                            "⬇  Download Annotated Image",
                            data=buf.getvalue(),
                            file_name=f"leafscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True,
                        )

                        if dets:
                            st.markdown("<br>", unsafe_allow_html=True)
                            for d in sorted(dets, key=lambda x: x["conf"], reverse=True):
                                render_single_result(d["name"], d["conf"])
                                st.markdown("<br>", unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='text-align:center;padding:40px;
                                        color:var(--muted);font-size:0.9rem;
                                        background:var(--cream);border-radius:10px;
                                        border:1px dashed var(--border);'>
                                No detections above threshold.<br>
                                Try lowering the confidence threshold in the sidebar.
                            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:var(--muted);
                        font-size:0.9rem;'>
                Upload an apple leaf image to begin analysis
            </div>""", unsafe_allow_html=True)

    # Right panel when no upload yet
    if not uploaded_img:
        with res_col:
            st.markdown("""
            <div style='background:var(--cream);border:1px solid var(--border);
                        border-radius:10px;padding:32px;'>
                <div style='font-family:"Cormorant Garamond",serif;font-size:1.3rem;
                            font-weight:600;color:var(--forest);margin-bottom:16px;'>
                    How It Works
                </div>
                <div style='display:flex;flex-direction:column;gap:16px;'>
                    <div style='display:flex;gap:14px;align-items:flex-start;'>
                        <span style='font-size:1.5rem;'>📁</span>
                        <div>
                            <div style='font-weight:500;color:var(--text);margin-bottom:3px;'>1. Upload Image</div>
                            <div style='font-size:0.82rem;color:var(--text2);'>
                                Upload a JPG or PNG photo of an apple leaf (single leaf or branch).
                            </div>
                        </div>
                    </div>
                    <div style='display:flex;gap:14px;align-items:flex-start;'>
                        <span style='font-size:1.5rem;'>🔬</span>
                        <div>
                            <div style='font-weight:500;color:var(--text);margin-bottom:3px;'>2. Run Analysis</div>
                            <div style='font-size:0.82rem;color:var(--text2);'>
                                YOLOv8 scans the image and localises disease regions.
                            </div>
                        </div>
                    </div>
                    <div style='display:flex;gap:14px;align-items:flex-start;'>
                        <span style='font-size:1.5rem;'>📊</span>
                        <div>
                            <div style='font-weight:500;color:var(--text);margin-bottom:3px;'>3. Get Diagnosis</div>
                            <div style='font-size:0.82rem;color:var(--text2);'>
                                Receive disease ID, confidence score, severity, and treatment guidance.
                            </div>
                        </div>
                    </div>
                    <div style='display:flex;gap:14px;align-items:flex-start;'>
                        <span style='font-size:1.5rem;'>⬇</span>
                        <div>
                            <div style='font-weight:500;color:var(--text);margin-bottom:3px;'>4. Download Report</div>
                            <div style='font-size:0.82rem;color:var(--text2);'>
                                Save the annotated image with bounding boxes for your records.
                            </div>
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — LIVE CAMERA
# ══════════════════════════════════════════════
with tab_camera:
    st.markdown("<br>", unsafe_allow_html=True)

    # Controls
    ctrl1, ctrl2, ctrl3, _ = st.columns([1,1,1,4])
    start_cam = ctrl1.button("▶  Start Camera", use_container_width=True)
    stop_cam  = ctrl2.button("■  Stop",         use_container_width=True)
    clr_cam   = ctrl3.button("↺  Reset",        use_container_width=True)

    if start_cam:
        ss["cam_running"] = True
        ss["frame_count"] = 0
    if stop_cam:
        ss["cam_running"] = False
    if clr_cam:
        ss["cam_running"] = False
        ss["frame_count"] = 0
        ss["fps"] = 0.0
        ss["last_dets"] = []

    # Layout
    cam_col, info_col = st.columns([2.2, 1], gap="large")

    with cam_col:
        # Status
        status_ph = st.empty()
        if ss["cam_running"]:
            status_ph.markdown(
                '<div class="live-dot"><span class="dot dot-live"></span> LIVE DETECTION</div>',
                unsafe_allow_html=True,
            )
        else:
            status_ph.markdown(
                '<div class="live-dot"><span class="dot dot-idle"></span> CAMERA IDLE</div>',
                unsafe_allow_html=True,
            )

        frame_ph = st.empty()
        frame_ph.markdown("""
        <div style='background:#1a2a1a;border:1px solid #2d5a3d;border-radius:10px;
                    height:420px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:12px;'>
            <div style='font-size:4rem;opacity:0.2;'>🎥</div>
            <div style='font-family:"DM Mono",monospace;font-size:0.68rem;
                        color:rgba(82,183,136,0.3);letter-spacing:2px;text-transform:uppercase;'>
                Press START CAMERA to begin
            </div>
        </div>""", unsafe_allow_html=True)

    with info_col:
        # Live metrics
        st.markdown("""
        <div style='font-family:"DM Mono",monospace;font-size:0.62rem;text-transform:uppercase;
                    letter-spacing:2px;color:var(--muted);margin-bottom:10px;'>
            Live Metrics
        </div>""", unsafe_allow_html=True)

        fps_ph   = st.empty()
        det_ph   = st.empty()
        frame_ph2 = st.empty()

        st.markdown("""
        <div style='font-family:"DM Mono",monospace;font-size:0.62rem;text-transform:uppercase;
                    letter-spacing:2px;color:var(--muted);margin:16px 0 10px;'>
            Current Detections
        </div>""", unsafe_allow_html=True)

        det_list_ph = st.empty()
        det_list_ph.markdown(
            '<div style="color:var(--muted);font-size:0.8rem;padding:10px 0;">'
            'No detections yet…</div>', unsafe_allow_html=True
        )

    def update_cam_metrics():
        fps_ph.markdown(f"""
        <div class="chip" style="margin-bottom:8px;">
            <div class="chip-val">{ss['fps']:.1f}</div>
            <div class="chip-lbl">FPS</div>
        </div>""", unsafe_allow_html=True)
        det_ph.markdown(f"""
        <div class="chip" style="margin-bottom:8px;">
            <div class="chip-val">{len(ss['last_dets'])}</div>
            <div class="chip-lbl">Detections</div>
        </div>""", unsafe_allow_html=True)
        frame_ph2.markdown(f"""
        <div class="chip" style="margin-bottom:8px;">
            <div class="chip-val">{ss['frame_count']}</div>
            <div class="chip-lbl">Frames</div>
        </div>""", unsafe_allow_html=True)

    update_cam_metrics()

    # ── Camera loop ───────────────────────────
    if ss["cam_running"]:
        if not Path(model_path).exists():
            status_ph.markdown(
                '<div class="live-dot"><span class="dot dot-error"></span> MODEL NOT FOUND</div>',
                unsafe_allow_html=True,
            )
            st.error(f"Model file not found: `{model_path}`")
        else:
            model, err = load_yolo(model_path)
            if err:
                st.error(f"Model load error: {err}")
            else:
                cap = cv2.VideoCapture(cam_idx)
                if not cap.isOpened():
                    status_ph.markdown(
                        '<div class="live-dot"><span class="dot dot-error"></span> CAMERA ERROR</div>',
                        unsafe_allow_html=True,
                    )
                    st.error(f"Cannot open camera {cam_idx}")
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                    t_start = time.time()
                    frame_interval = 1.0 / max_fps

                    try:
                        while ss["cam_running"]:
                            t_loop = time.time()
                            ret, frame = cap.read()
                            if not ret:
                                time.sleep(0.05)
                                continue

                            results = model.predict(
                                frame, conf=conf_thresh, iou=iou_thresh,
                                imgsz=img_size, verbose=False
                            )

                            annotated, dets = annotate_image(
                                frame, results, model,
                                show_labels, show_conf, BOX_COLOR
                            )

                            ss["last_dets"] = dets
                            ss["frame_count"] += 1
                            elapsed_total = time.time() - t_start
                            ss["fps"] = ss["frame_count"] / elapsed_total if elapsed_total > 0 else 0

                            # Overlay HUD
                            h, w = annotated.shape[:2]
                            overlay = annotated.copy()
                            cv2.rectangle(overlay, (0, h-36), (w, h), (26,42,26), -1)
                            annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
                            ts = datetime.now().strftime("%H:%M:%S")
                            cv2.putText(annotated,
                                f"LeafScan  |  {ts}  |  {ss['fps']:.1f} fps  |  {len(dets)} det",
                                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (180,220,180), 1, cv2.LINE_AA)

                            img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_ph.image(img_rgb, channels="RGB", use_container_width=True)

                            update_cam_metrics()

                            # Detection list
                            if dets:
                                html = "".join(
                                    f'<div class="det-row">'
                                    f'<span style="font-size:1.1rem;">{d["icon"]}</span>'
                                    f'<span class="det-name">{d["display"]}</span>'
                                    f'<span class="det-conf-pill" style="background:{d["bg"]};color:{d["color"]};">'
                                    f'{d["conf"]*100:.1f}%</span></div>'
                                    for d in sorted(dets, key=lambda x: x["conf"], reverse=True)
                                )
                                det_list_ph.markdown(html, unsafe_allow_html=True)

                                # Add to history
                                for d in dets:
                                    if (not ss["history"] or
                                            ss["history"][0].get("disease") != d["display"] or
                                            time.time() % 3 < 0.1):
                                        ss["history"].insert(0, {
                                            "time": datetime.now().strftime("%H:%M:%S"),
                                            "disease": d["display"],
                                            "conf": d["conf"],
                                            "icon": d["icon"],
                                            "source": "camera",
                                        })
                                ss["history"] = ss["history"][:100]
                            else:
                                det_list_ph.markdown(
                                    '<div style="color:var(--muted);font-size:0.8rem;'
                                    'padding:10px 0;">No detections</div>', unsafe_allow_html=True
                                )

                            # Throttle
                            sleep_t = frame_interval - (time.time() - t_loop)
                            if sleep_t > 0:
                                time.sleep(sleep_t)

                    finally:
                        cap.release()
                        status_ph.markdown(
                            '<div class="live-dot"><span class="dot dot-idle"></span> STOPPED</div>',
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════
with tab_history:
    st.markdown("<br>", unsafe_allow_html=True)

    h_col, stats_col = st.columns([2, 1], gap="large")

    with h_col:
        st.markdown("""
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.5rem;
                    font-weight:600;color:var(--forest);margin-bottom:14px;'>
            Detection History
        </div>""", unsafe_allow_html=True)

        if st.button("🗑  Clear History"):
            ss["history"] = []

        if ss["history"]:
            for h in ss["history"][:40]:
                conf_col = "#27ae60" if h["conf"]>=0.75 else "#f39c12" if h["conf"]>=0.5 else "#e74c3c"
                src_icon = "📷" if h["source"]=="upload" else "🎥"
                st.markdown(f"""
                <div class="hist-item">
                    <span class="hist-time">{h["time"]}</span>
                    <span style="font-size:1.1rem;">{h["icon"]}</span>
                    <span style="font-weight:500;color:var(--text);flex:1;">{h["disease"]}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.7rem;
                                 padding:2px 10px;border-radius:20px;
                                 background:rgba(0,0,0,0.05);color:{conf_col};">
                        {h["conf"]*100:.1f}%
                    </span>
                    <span style="color:var(--muted);font-size:0.9rem;margin-left:6px;">{src_icon}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px;color:var(--muted);
                        background:var(--cream);border-radius:10px;
                        border:1px dashed var(--border);font-size:0.9rem;'>
                No detections recorded yet.<br>
                Use Image Upload or Live Camera to begin.
            </div>""", unsafe_allow_html=True)

    with stats_col:
        st.markdown("""
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.5rem;
                    font-weight:600;color:var(--forest);margin-bottom:14px;'>
            Summary
        </div>""", unsafe_allow_html=True)

        if ss["history"]:
            # Count by disease
            counts = {}
            confs  = {}
            for h in ss["history"]:
                d = h["disease"]
                counts[d] = counts.get(d, 0) + 1
                confs.setdefault(d, []).append(h["conf"])

            st.markdown(f"""
            <div class="chip-row" style="grid-template-columns:1fr 1fr;">
                <div class="chip">
                    <div class="chip-val">{len(ss["history"])}</div>
                    <div class="chip-lbl">Total</div>
                </div>
                <div class="chip">
                    <div class="chip-val">{len(counts)}</div>
                    <div class="chip-lbl">Classes</div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style='font-family:"DM Mono",monospace;font-size:0.62rem;text-transform:uppercase;
                        letter-spacing:2px;color:var(--muted);margin:16px 0 10px;'>
                By Disease
            </div>""", unsafe_allow_html=True)

            total = sum(counts.values())
            for disease, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                pct = cnt / total * 100
                info = get_disease_info(disease)
                avg_conf = sum(confs[disease]) / len(confs[disease])
                st.markdown(f"""
                <div style='margin-bottom:14px;'>
                    <div style='display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:5px;'>
                        <span style='font-size:0.85rem;font-weight:500;color:var(--text);'>
                            {info["icon"]} {disease}
                        </span>
                        <span style='font-family:"DM Mono",monospace;font-size:0.7rem;
                                     color:var(--muted);'>{cnt}x</span>
                    </div>
                    <div class="sev-bar-wrap">
                        <div class="sev-bar-fill" style="width:{pct}%;background:{info['color']};"></div>
                    </div>
                    <div style='font-family:"DM Mono",monospace;font-size:0.6rem;
                                color:var(--muted);'>avg conf: {avg_conf*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Export JSON
            export_data = json.dumps(ss["history"], indent=2)
            st.download_button(
                "⬇  Export History (JSON)",
                data=export_data,
                file_name=f"leafscan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style='color:var(--muted);font-size:0.85rem;text-align:center;padding:20px;'>
                Statistics will appear after detections
            </div>""", unsafe_allow_html=True)