"""
EduPredict — Student Performance Intelligence Platform
======================================================
Run:
    streamlit run app.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from data_generator import save_dataset
from predictor import load_artifacts, predict
from train_model import FEATURES

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict — Student Performance Intelligence",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN SYSTEM  (CSS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Outfit:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');

/* ── Design Tokens ───────────────────────────────────────────────── */
:root {
    --bg-root:        #070b12;
    --bg-surface:     #0c1220;
    --bg-card:        #101928;
    --bg-card-hover:  #131f32;
    --bg-input:       #0e1624;

    --indigo:         #6366f1;
    --indigo-dim:     rgba(99,102,241,0.12);
    --indigo-border:  rgba(99,102,241,0.25);
    --amber:          #f59e0b;
    --amber-dim:      rgba(245,158,11,0.10);
    --amber-border:   rgba(245,158,11,0.25);
    --emerald:        #10b981;
    --emerald-dim:    rgba(16,185,129,0.10);
    --rose:           #f43f5e;
    --rose-dim:       rgba(244,63,94,0.10);
    --sky:            #38bdf8;

    --text-primary:   #f0f4ff;
    --text-secondary: #8892a4;
    --text-muted:     #4a5568;
    --border:         rgba(255,255,255,0.06);
    --border-accent:  rgba(99,102,241,0.20);

    --font-display:   'Cormorant Garamond', Georgia, serif;
    --font-body:      'Outfit', sans-serif;
    --font-mono:      'Fira Code', monospace;

    --radius-sm:  8px;
    --radius-md:  12px;
    --radius-lg:  18px;
    --radius-xl:  24px;
}

/* ── Base ────────────────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-root) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 0%, rgba(99,102,241,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 100%, rgba(245,158,11,0.05) 0%, transparent 55%);
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}
*, *::before, *::after { box-sizing: border-box; }

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-root); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--indigo); }

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── Streamlit UI Overrides ──────────────────────────────────────── */
.stSlider label, .stRadio label, .stSelectbox label {
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.02em !important;
}
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
.stButton button {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: var(--radius-sm) !important;
    letter-spacing: 0.03em !important;
    transition: all 0.18s ease !important;
    text-transform: uppercase !important;
}
.stButton button[kind="primary"] {
    background: var(--indigo) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.35) !important;
}
.stButton button[kind="primary"]:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important;
}
.stButton button:not([kind="primary"]) {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
}
.stButton button:not([kind="primary"]):hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--indigo-border) !important;
    color: var(--text-primary) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--indigo-dim) !important;
    color: var(--indigo) !important;
    border-bottom: 2px solid var(--indigo) !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}
.stDataFrame {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-mono) !important;
}
.stAlert {
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
}
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── COMPONENTS ──────────────────────────────────────────────────── */

/* Wordmark */
.wordmark {
    font-family: var(--font-display);
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    line-height: 1;
}
.wordmark span { color: var(--indigo); }

/* Nav item */
.nav-section-label {
    font-family: var(--font-body);
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 0 20px;
    margin-bottom: 6px;
}

/* Stat badge in sidebar */
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
}
.sidebar-stat-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-family: var(--font-body);
}
.sidebar-stat-value {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 600;
}

/* Hero */
.hero {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 56px 52px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%; left: -10%;
    width: 55%; height: 200%;
    background: radial-gradient(ellipse, rgba(99,102,241,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -50%; right: -5%;
    width: 45%; height: 170%;
    background: radial-gradient(ellipse, rgba(245,158,11,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--indigo);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-label::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1px;
    background: var(--indigo);
}
.hero-title {
    font-family: var(--font-display) !important;
    font-size: 3.8rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    line-height: 1.05 !important;
    margin: 0 0 20px 0 !important;
    letter-spacing: -0.02em !important;
}
.hero-title em {
    font-style: italic;
    color: var(--indigo);
}
.hero-desc {
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 300;
    max-width: 520px;
    line-height: 1.75;
    margin: 0;
}
.hero-tag {
    display: inline-block;
    background: var(--indigo-dim);
    border: 1px solid var(--indigo-border);
    color: var(--indigo);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 4px;
    margin: 4px 3px 0 0;
}

/* Metric card */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 22px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.kpi-card:hover {
    border-color: var(--indigo-border);
    transform: translateY(-2px);
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--indigo), var(--amber));
    opacity: 0.7;
}
.kpi-icon {
    width: 36px; height: 36px;
    background: var(--indigo-dim);
    border: 1px solid var(--indigo-border);
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 16px;
}
.kpi-value {
    font-family: var(--font-mono) !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    line-height: 1 !important;
    margin-bottom: 6px !important;
}
.kpi-label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Section heading */
.sec-heading {
    font-family: var(--font-display);
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin-bottom: 4px;
}
.sec-rule {
    width: 40px; height: 2px;
    background: var(--indigo);
    border-radius: 2px;
    margin-bottom: 24px;
}
.sec-sub {
    font-size: 0.88rem;
    color: var(--text-secondary);
    font-weight: 300;
    margin-top: -16px;
    margin-bottom: 24px;
    line-height: 1.6;
}

/* Panel */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 24px;
}
.panel-title {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}

/* Result display */
.result-marks-display {
    font-family: var(--font-mono) !important;
    font-size: 5rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    line-height: 1 !important;
}
.result-marks-display sup {
    font-size: 1.4rem;
    color: var(--text-muted);
    font-weight: 400;
    vertical-align: super;
    margin-left: 4px;
}
.result-tag {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 4px;
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.tag-excellent { background: var(--emerald-dim); border: 1px solid rgba(16,185,129,0.3); color: var(--emerald); }
.tag-good      { background: var(--indigo-dim);  border: 1px solid var(--indigo-border); color: var(--indigo); }
.tag-average   { background: var(--amber-dim);   border: 1px solid var(--amber-border);  color: var(--amber); }
.tag-poor      { background: var(--rose-dim);    border: 1px solid rgba(244,63,94,0.3);  color: var(--rose); }

/* Progress bar */
.progress-track {
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    height: 4px;
    margin: 16px 0;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--indigo), var(--amber));
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* Mini stat */
.mini-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 12px;
    text-align: center;
}
.mini-stat-val {
    font-family: var(--font-mono) !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}
.mini-stat-lbl {
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-top: 4px !important;
}

/* Tip row */
.tip-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 14px 16px;
    background: rgba(99,102,241,0.04);
    border: 1px solid var(--border);
    border-left: 3px solid var(--indigo);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin-bottom: 10px;
    font-family: var(--font-body);
    font-size: 0.86rem;
    color: var(--text-secondary);
    line-height: 1.65;
}
.tip-dot {
    width: 6px; height: 6px;
    background: var(--indigo);
    border-radius: 50%;
    margin-top: 7px;
    flex-shrink: 0;
}

/* About card */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 22px;
    height: 100%;
}
.info-card-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--indigo);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}
.info-card-title {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
}
.info-card-body {
    font-size: 0.86rem;
    color: var(--text-secondary);
    line-height: 1.75;
    font-weight: 300;
}

/* Step number */
.step-num {
    font-family: var(--font-mono);
    font-size: 2.8rem;
    font-weight: 600;
    color: rgba(99,102,241,0.18);
    line-height: 1;
    margin-bottom: 10px;
}

/* Code block */
.code-block {
    background: rgba(0,0,0,0.35);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--amber);
    line-height: 1.7;
    margin: 12px 0;
}

/* Footer */
.footer {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 36px;
}
.footer-brand {
    font-family: var(--font-display);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
}
.footer-copy {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 2px;
    font-family: var(--font-body);
}
.pill {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 3px 9px;
    border-radius: 4px;
}

/* Separator line with label */
.sep-label {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 24px 0;
}
.sep-line { flex: 1; height: 1px; background: var(--border); }
.sep-text {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    text-align: center;
    gap: 12px;
}
.empty-icon {
    width: 52px; height: 52px;
    background: var(--indigo-dim);
    border: 1px solid var(--indigo-border);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    path = "data/students.csv"
    if not os.path.exists(path):
        save_dataset(path)
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


def reset_inputs():
    st.session_state.update(
        study_hours=5.0, attendance=75.0,
        sleep_hours=7.0, assignments=7, stress=5
    )


def load_example():
    st.session_state.update(
        study_hours=8.0, attendance=92.0,
        sleep_hours=7.5, assignments=9, stress=3
    )


for k, v in [("study_hours", 5.0), ("attendance", 75.0), ("sleep_hours", 7.0),
             ("assignments", 7), ("stress", 5), ("page", "Overview"),
             ("last_result", None)]:
    st.session_state.setdefault(k, v)

reg, clf, scaler, metrics = get_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
#  CHART THEME
# ─────────────────────────────────────────────────────────────────────────────
CHART_BG = "#101928"
CHART_GRID = "#1a2540"

plt.rcParams.update({
    "figure.facecolor":  CHART_BG,
    "axes.facecolor":    CHART_BG,
    "axes.edgecolor":    CHART_GRID,
    "axes.labelcolor":   "#8892a4",
    "xtick.color":       "#8892a4",
    "ytick.color":       "#8892a4",
    "text.color":        "#f0f4ff",
    "grid.color":        CHART_GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "axes.grid":         True,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

C_INDIGO  = "#6366f1"
C_AMBER   = "#f59e0b"
C_EMERALD = "#10b981"
C_ROSE    = "#f43f5e"
C_SKY     = "#38bdf8"
PERF_COLORS = {
    "Poor": C_ROSE, "Average": C_AMBER,
    "Good": C_INDIGO, "Excellent": C_EMERALD
}


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 28px 20px 24px;">
        <div class="wordmark">Edu<span>Predict</span></div>
        <div style="font-family:'Outfit',sans-serif; font-size:0.72rem;
                    color:#4a5568; margin-top:5px; letter-spacing:0.04em;">
            Performance Intelligence Platform
        </div>
    </div>
    <div style="height:1px; background:var(--border); margin: 0 0 16px 0;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Overview", "Predictor", "Analytics", "About"],
        label_visibility="collapsed",
        key="page",
    )

    st.markdown("""
    <div style="height:1px; background:var(--border); margin: 16px 0;"></div>
    <div class="nav-section-label">Model Performance</div>
    """, unsafe_allow_html=True)

    stat_color = {"acc": C_EMERALD, "r2": C_INDIGO, "mae": C_AMBER}
    st.markdown(f"""
    <div style="padding: 0 4px;">
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Classifier Accuracy</span>
            <span class="sidebar-stat-value" style="color:{stat_color['acc']};">
                {metrics['classification_accuracy']*100:.1f}%</span>
        </div>
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Regression R²</span>
            <span class="sidebar-stat-value" style="color:{stat_color['r2']};">
                {metrics['r2_score']:.4f}</span>
        </div>
        <div class="sidebar-stat" style="border-bottom:none;">
            <span class="sidebar-stat-label">Mean Abs. Error</span>
            <span class="sidebar-stat-value" style="color:{stat_color['mae']};">
                {metrics['mae']:.2f} pts</span>
        </div>
    </div>
    <div style="height:1px; background:var(--border); margin: 16px 0;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 0 4px 24px;">
        <div class="nav-section-label" style="margin-bottom:10px;">Stack</div>
        <div style="display:flex; flex-wrap:wrap; gap:5px;">
            <span class="pill">scikit-learn</span>
            <span class="pill">streamlit</span>
            <span class="pill">pandas</span>
            <span class="pill">seaborn</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str, sub: str = ""):
    st.markdown(f'<div class="sec-heading">{title}</div>'
                f'<div class="sec-rule"></div>'
                + (f'<div class="sec-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE  1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    df = get_dataset()

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">Machine Learning  ·  Academic Intelligence</div>
        <div class="hero-title">Predict Student<br><em>Academic Performance</em></div>
        <p class="hero-desc">
            Input five behavioral indicators and receive a data-driven prediction of
            final marks, pass probability, and performance classification —
            powered by Linear and Logistic Regression.
        </p>
        <div style="margin-top:24px;">
            <span class="hero-tag">Linear Regression</span>
            <span class="hero-tag">Logistic Regression</span>
            <span class="hero-tag">StandardScaler</span>
            <span class="hero-tag">Synthetic Dataset</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    pass_rate = df['pass_fail'].mean() * 100 if 'pass_fail' in df.columns else (df['marks'] >= 40).mean() * 100

    kpi_data = [
        ("Classifier Accuracy", f"{metrics['classification_accuracy']*100:.1f}%",
         "Logistic Regression", C_EMERALD),
        ("R² Score", f"{metrics['r2_score']:.4f}",
         "Linear Regression fit", C_INDIGO),
        ("Mean Abs. Error", f"{metrics['mae']:.2f}",
         "Marks deviation", C_AMBER),
        ("Training Samples", f"{len(df):,}",
         "Synthetic records", C_SKY),
    ]
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val, sub_lbl, color) in zip([c1, c2, c3, c4], kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <rect x="1" y="8" width="3" height="7" rx="1" fill="{color}" opacity="0.7"/>
                        <rect x="6" y="4" width="3" height="11" rx="1" fill="{color}"/>
                        <rect x="11" y="1" width="3" height="14" rx="1" fill="{color}" opacity="0.5"/>
                    </svg>
                </div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
                <div style="font-size:0.7rem; color:var(--text-muted); margin-top:4px;
                            font-family:var(--font-mono);">{sub_lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("How It Works", "A three-stage pipeline from input to actionable prediction")

    s1, s2, s3 = st.columns(3)
    steps = [
        (s1, "01", "Data Input", "Provide five quantitative indicators — study hours, attendance, sleep, assignment completion, and stress level."),
        (s2, "02", "Model Inference", "Two models run in parallel: Linear Regression outputs a continuous marks score; Logistic Regression outputs a pass probability."),
        (s3, "03", "Result Analysis", "Receive a structured report with predicted marks, performance tier, outcome classification, and personalized recommendations."),
    ]
    for col, num, title, desc in steps:
        with col:
            st.markdown(f"""
            <div class="info-card">
                <div class="step-num">{num}</div>
                <div class="info-card-title">{title}</div>
                <div class="info-card-body">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Sample Data", "First eight records from the training dataset")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:var(--indigo-dim); border:1px solid var(--indigo-border);
                border-radius:var(--radius-md); padding:14px 18px; font-size:0.86rem;
                color:var(--text-secondary); font-family:var(--font-body);
                margin-top:16px;">
        Navigate to <strong style="color:var(--indigo);">Predictor</strong> in the sidebar
        to run a live prediction, or open <strong style="color:var(--indigo);">Analytics</strong>
        to explore the full dataset.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE  2 — PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Predictor":
    section("Predictor", "Configure the student profile and run inference")

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown('<div class="panel"><div class="panel-title">Student Profile</div>',
                    unsafe_allow_html=True)
        st.slider("Study Hours per Day", 0.0, 12.0, key="study_hours", step=0.5)
        st.slider("Attendance (%)", 0.0, 100.0, key="attendance", step=1.0)
        st.slider("Sleep Hours per Day", 3.0, 10.0, key="sleep_hours", step=0.5)
        st.slider("Assignments Completed (out of 10)", 0, 10, key="assignments")
        st.slider("Stress Level  (1 = low  /  10 = high)", 1, 10, key="stress")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        predict_clicked = b1.button("Run Prediction", type="primary", use_container_width=True)
        b2.button("Load Example", on_click=load_example, use_container_width=True)
        b3.button("Reset", on_click=reset_inputs, use_container_width=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Inference Output</div>',
                    unsafe_allow_html=True)

        if predict_clicked:
            with st.spinner("Running models..."):
                result = predict(
                    study_hours=st.session_state["study_hours"],
                    attendance=st.session_state["attendance"],
                    sleep_hours=st.session_state["sleep_hours"],
                    assignments_completed=st.session_state["assignments"],
                    stress_level=st.session_state["stress"],
                )
            st.session_state["last_result"] = result

        res = st.session_state["last_result"]

        if res is not None:
            perf = res.performance_label.lower()
            tag_cls = f"tag-{perf}"
            outcome_color = C_EMERALD if res.will_pass else C_ROSE
            outcome_text  = "PASS" if res.will_pass else "FAIL"
            pct           = int(res.predicted_marks)

            r1, r2 = st.columns([1, 1])
            with r1:
                st.markdown(f"""
                <div>
                    <div style="font-family:var(--font-mono); font-size:0.65rem;
                                color:var(--text-muted); text-transform:uppercase;
                                letter-spacing:0.1em; margin-bottom:8px;">Predicted Marks</div>
                    <div class="result-marks-display">{res.predicted_marks:.1f}<sup>/ 100</sup></div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div style="padding-top:10px;">
                    <div style="margin-bottom:14px;">
                        <span class="result-tag {tag_cls}">{res.performance_label}</span>
                    </div>
                    <div style="font-family:var(--font-mono); font-size:1.5rem;
                                font-weight:600; color:{outcome_color};">{outcome_text}</div>
                    <div style="font-family:var(--font-body); font-size:0.72rem;
                                color:var(--text-muted); margin-top:2px; text-transform:uppercase;
                                letter-spacing:0.08em;">Outcome</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct}%;"></div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="mini-stat">
                    <div class="mini-stat-val">{res.predicted_marks:.1f}</div>
                    <div class="mini-stat-lbl">Score</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="mini-stat">
                    <div class="mini-stat-val">{res.pass_probability*100:.1f}%</div>
                    <div class="mini-stat-lbl">Pass Probability</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="mini-stat">
                    <div class="mini-stat-val" style="color:{outcome_color};">
                        {outcome_text}</div>
                    <div class="mini-stat-lbl">Classification</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(res.explanation)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">
                    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                        <circle cx="11" cy="11" r="9" stroke="#6366f1" stroke-width="1.5"/>
                        <path d="M11 7v5M11 15v.5" stroke="#6366f1" stroke-width="1.5"
                              stroke-linecap="round"/>
                    </svg>
                </div>
                <div style="font-size:0.9rem; color:var(--text-secondary);
                            font-family:var(--font-body); line-height:1.7;">
                    Configure a student profile on the left<br>and click
                    <strong style="color:var(--indigo);">Run Prediction</strong>.
                </div>
                <div style="font-size:0.75rem; color:var(--text-muted);
                            font-family:var(--font-mono);">
                    Use "Load Example" for a demo.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Study Tips
    if st.session_state["last_result"] is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        section("Recommendations", "Actionable insights based on the current profile")

        tips = []
        if st.session_state["study_hours"] < 4:
            tips.append("Increase daily study time to at least 4–6 hours for consistent retention and exam readiness.")
        if st.session_state["attendance"] < 75:
            tips.append("Attendance below 75% significantly reduces exposure to core material. Target 85%+ each term.")
        if st.session_state["sleep_hours"] < 6:
            tips.append("Sleep deprivation impairs memory consolidation. Aim for 7–8 hours of uninterrupted sleep nightly.")
        if st.session_state["assignments"] < 7:
            tips.append("Completing assignments reinforces concepts and contributes directly to final grade. Target 9–10 out of 10.")
        if st.session_state["stress"] > 7:
            tips.append("Elevated stress reduces cognitive performance. Structured breaks, exercise, and time management can help.")
        if not tips:
            tips.append("Strong profile. Maintain current study habits consistently to secure top-tier performance.")

        t1, t2 = st.columns(2)
        for i, tip in enumerate(tips):
            with (t1 if i % 2 == 0 else t2):
                st.markdown(f"""
                <div class="tip-row">
                    <div class="tip-dot"></div>
                    <div>{tip}</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE  3 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Analytics":
    df = get_dataset()
    section("Analytics", "Exploratory analysis of the training dataset")

    pass_rate = df['pass_fail'].mean() * 100 if 'pass_fail' in df.columns else (df['marks'] >= 40).mean() * 100
    exc_rate  = (df['performance'] == 'Excellent').mean() * 100 if 'performance' in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, color in [
        (c1, f"{len(df):,}", "Total Records", C_INDIGO),
        (c2, f"{df['marks'].mean():.1f}", "Mean Score", C_AMBER),
        (c3, f"{pass_rate:.1f}%", "Pass Rate", C_EMERALD),
        (c4, f"{exc_rate:.1f}%", "Excellent Tier", C_SKY),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{color}; font-size:1.7rem;">{val}</div>
                <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Score vs Study",
        "Correlation Matrix",
        "Feature Coefficients",
        "Distributions",
        "Class Split",
    ])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for label, grp in df.groupby("performance"):
            axes[0].scatter(grp["study_hours"], grp["marks"],
                            label=label, color=PERF_COLORS.get(label, C_INDIGO),
                            alpha=0.55, s=18, zorder=3)
        z = np.polyfit(df["study_hours"], df["marks"], 1)
        xl = np.linspace(0, 12, 200)
        axes[0].plot(xl, np.poly1d(z)(xl), color="white", lw=1.5, ls="--", alpha=0.6)
        axes[0].set_title("Study Hours vs Final Score", fontsize=12, pad=12, color="#f0f4ff")
        axes[0].set_xlabel("Study Hours / Day")
        axes[0].set_ylabel("Final Marks")
        axes[0].legend(fontsize=8, framealpha=0.2)

        sc = axes[1].scatter(df["attendance"], df["marks"],
                             c=df["marks"], cmap="plasma", alpha=0.45, s=16, zorder=3)
        z2 = np.polyfit(df["attendance"], df["marks"], 1)
        xl2 = np.linspace(40, 100, 200)
        axes[1].plot(xl2, np.poly1d(z2)(xl2), color="white", lw=1.5, ls="--", alpha=0.6)
        axes[1].set_title("Attendance vs Final Score", fontsize=12, pad=12, color="#f0f4ff")
        axes[1].set_xlabel("Attendance (%)")
        axes[1].set_ylabel("Final Marks")
        fig.colorbar(sc, ax=axes[1], label="Marks", pad=0.01)
        fig.tight_layout(pad=3)
        st.pyplot(fig, clear_figure=True)

    with tab2:
        corr = df[FEATURES + ["marks"]].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.4, linecolor=CHART_GRID,
                    vmin=-1, vmax=1, square=True,
                    annot_kws={"size": 10, "weight": "bold", "color": "white"},
                    cbar_kws={"shrink": 0.75})
        ax.set_title("Feature Correlation Matrix", fontsize=13, pad=14, color="#f0f4ff")
        ax.tick_params(colors="#8892a4")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Values close to +1 indicate strong positive correlation with final marks.")

    with tab3:
        imp = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": reg.coef_,
        }).sort_values("Coefficient", key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bar_colors = [C_EMERALD if v >= 0 else C_ROSE for v in imp["Coefficient"]]
        bars = ax.barh(imp["Feature"], imp["Coefficient"],
                       color=bar_colors, edgecolor=CHART_BG, height=0.5, zorder=3)
        ax.axvline(0, color="white", lw=0.8, alpha=0.3)
        for bar, val in zip(bars, imp["Coefficient"]):
            ax.text(val + (0.15 if val >= 0 else -0.15),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8.5, color="white",
                    fontfamily="monospace")
        ax.set_title("Standardized Regression Coefficients", fontsize=12, pad=12, color="#f0f4ff")
        ax.set_xlabel("Coefficient Value")
        lim = max(abs(imp["Coefficient"].max()), abs(imp["Coefficient"].min())) + 1.5
        ax.set_xlim(-lim, lim)
        pos_p = mpatches.Patch(color=C_EMERALD, label="Positive impact")
        neg_p = mpatches.Patch(color=C_ROSE,    label="Negative impact")
        ax.legend(handles=[pos_p, neg_p], fontsize=8, framealpha=0.15)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.dataframe(
            imp.sort_values("Coefficient", key=abs, ascending=False)
               .reset_index(drop=True)
               .rename(columns={"Coefficient": "Coeff (Standardized)"}),
            use_container_width=True, hide_index=True
        )

    with tab4:
        all_cols = FEATURES + ["marks"]
        palette_dist = [C_INDIGO, C_AMBER, C_EMERALD, "#a78bfa", C_ROSE, C_SKY]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        for i, (c, col_c) in enumerate(zip(all_cols, palette_dist)):
            axes[i].hist(df[c], bins=25, color=col_c, alpha=0.8,
                         edgecolor=CHART_BG, zorder=3)
            axes[i].axvline(df[c].mean(), color="white", ls="--", lw=1.3, alpha=0.7)
            axes[i].set_title(c.replace("_", " ").title(), fontsize=11,
                              pad=10, color="#f0f4ff")
            axes[i].text(0.97, 0.94, f"μ={df[c].mean():.1f}",
                         transform=axes[i].transAxes, ha="right", va="top",
                         fontsize=8, color="white", fontfamily="monospace",
                         alpha=0.7)
        fig.suptitle("Feature Distributions", fontsize=14, y=1.01, color="#f0f4ff")
        fig.tight_layout(pad=2.5)
        st.pyplot(fig, clear_figure=True)

    with tab5:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        order = ["Poor", "Average", "Good", "Excellent"]
        bar_c = [PERF_COLORS[p] for p in order]
        counts = df["performance"].value_counts().reindex(order).fillna(0)
        axes[0].bar(order, counts.values, color=bar_c, edgecolor=CHART_BG,
                    width=0.5, zorder=3)
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 2, str(int(v)), ha="center", fontsize=10,
                         color="white", fontfamily="monospace")
        axes[0].set_title("Performance Tier Distribution", fontsize=12,
                          pad=12, color="#f0f4ff")
        axes[0].set_ylabel("Count")

        if "pass_fail" in df.columns:
            p_count = df["pass_fail"].sum()
            f_count = len(df) - p_count
        else:
            p_count = (df["marks"] >= 40).sum()
            f_count = (df["marks"] < 40).sum()
        axes[1].pie(
            [p_count, f_count], labels=["Pass", "Fail"],
            colors=[C_EMERALD, C_ROSE], autopct="%1.1f%%",
            startangle=90, pctdistance=0.78,
            wedgeprops={"edgecolor": CHART_BG, "linewidth": 2},
            textprops={"color": "white", "fontsize": 11}
        )
        axes[1].set_title("Pass / Fail Ratio", fontsize=12, pad=12, color="#f0f4ff")
        fig.tight_layout(pad=3)
        st.pyplot(fig, clear_figure=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE  4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About":
    section("About EduPredict", "Project overview, methodology, and dataset documentation")

    st.markdown(f"""
    <div class="hero" style="padding:38px 44px;">
        <div class="hero-label">Academic ML Project</div>
        <div class="hero-title" style="font-size:2.6rem;">EduPredict</div>
        <p class="hero-desc">
            A machine learning system that predicts student academic outcomes
            from five behavioral indicators. Combines a regression model for
            continuous score estimation with a classifier for binary
            pass/fail probability output.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section("Models")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-label">Regression</div>
            <div class="info-card-title">Linear Regression</div>
            <div class="info-card-body">
                Predicts the continuous marks value (0–100) a student is expected
                to score. Features are scaled using StandardScaler before inference.
                <br><br>
                <strong style="color:var(--text-primary);">Target —</strong>
                Final Marks (float)<br>
                <strong style="color:var(--text-primary);">Metrics —</strong>
                R² Score, MAE, RMSE
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-label">Classification</div>
            <div class="info-card-title">Logistic Regression</div>
            <div class="info-card-body">
                Predicts the probability of a student passing (marks ≥ 40).
                Returns a calibrated probability via sigmoid output.
                <br><br>
                <strong style="color:var(--text-primary);">Target —</strong>
                Pass / Fail (binary)<br>
                <strong style="color:var(--text-primary);">Metrics —</strong>
                Accuracy, Precision, Recall, F1
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Input Features")
    feat_df = pd.DataFrame({
        "Feature":     FEATURES,
        "Description": [
            "Average hours spent studying per day",
            "Percentage of classes attended",
            "Average hours of sleep per night",
            "Assignments submitted (out of 10)",
            "Self-reported stress level (1–10)",
        ],
        "Range":  ["0 – 12 hrs", "0 – 100%", "3 – 10 hrs", "0 – 10", "1 – 10"],
        "Effect": ["Positive", "Positive", "Positive", "Positive", "Negative"],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Dataset")
    st.markdown("""
    <div class="info-card" style="margin-bottom:16px;">
        <div class="info-card-label">Data Source</div>
        <div class="info-card-title">Synthetic Dataset</div>
        <div class="info-card-body">
            Generated using NumPy to simulate realistic student behavior.
            Marks are computed via a weighted linear formula with Gaussian noise.
        </div>
        <div class="code-block">
            marks = 5.5 × study_hours + 0.35 × attendance + 1.2 × sleep_hours<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            + 2.5 × assignments − 1.8 × stress_level + N(0, 5)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Real-world dataset alternatives:**")
    d1, d2, d3 = st.columns(3)
    for col, src, name, url in [
        (d1, "UCI ML Repository", "Student Performance Dataset",
         "https://archive.ics.uci.edu/dataset/320/student+performance"),
        (d2, "Kaggle", "Students Performance in Exams",
         "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams"),
        (d3, "Kaggle", "Student Habits & Academic Performance",
         "https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset"),
    ]:
        with col:
            st.markdown(f"""
            <div class="info-card" style="text-align:center; padding:20px;">
                <div class="info-card-label">{src}</div>
                <div class="info-card-title" style="font-size:0.95rem;">{name}</div>
                <a href="{url}" target="_blank"
                   style="font-family:var(--font-mono); font-size:0.72rem;
                          color:var(--indigo); text-decoration:none;">
                    View Dataset &rarr;</a>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Technology Stack")
    tech_items = [
        ("Python 3.10+",   "Core language"),
        ("scikit-learn",   "ML models · StandardScaler · metrics"),
        ("pandas",         "Dataframe operations"),
        ("NumPy",          "Numerical computation"),
        ("Streamlit",      "Web application framework"),
        ("matplotlib",     "Chart rendering"),
        ("seaborn",        "Statistical visualizations"),
        ("joblib",         "Model serialization"),
    ]
    t1, t2 = st.columns(2)
    for i, (name, desc) in enumerate(tech_items):
        with (t1 if i % 2 == 0 else t2):
            st.markdown(f"""
            <div class="mini-stat" style="text-align:left; margin-bottom:8px;
                         display:flex; align-items:center; gap:12px; padding:12px 14px;">
                <div>
                    <div style="font-family:var(--font-mono); font-size:0.82rem;
                                color:var(--text-primary); font-weight:600;">{name}</div>
                    <div style="font-size:0.72rem; color:var(--text-muted);
                                margin-top:2px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div>
        <div class="footer-brand">EduPredict</div>
        <div class="footer-copy">
            Student Performance Intelligence Platform &nbsp;·&nbsp;
            Synthetic dataset &nbsp;·&nbsp; Academic demonstration
        </div>
    </div>
    <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
        <span class="pill">scikit-learn</span>
        <span class="pill">streamlit</span>
        <span class="pill">Linear Regression</span>
        <span class="pill">Logistic Regression</span>
    </div>
</div>
""", unsafe_allow_html=True)