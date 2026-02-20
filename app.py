"""
app.py — OpenSIGAPS v5
Interface Streamlit — 3 onglets :
  🔬 Analyse individuelle  |  👥 Équipe / Service  |  📖 Méthodologie
"""

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backend import (
    C1_COEFFICIENTS,
    C2_COEFFICIENTS,
    CASCADE_LEVEL_LABELS,
    SIGAPS_MATRIX,
    VALEUR_POINT_EUROS,
    Article,
    FederatedSearch,  # alias de PubMedFetcher dans v7
    SigapsRefDB,
    calculate_fractional_score,
    calculate_presence_score,
    calculate_team_presence_score,
    fetch_article_by_doi,
    load_embed_model,
    search_journal_by_name,
    set_embed_model,
    set_ref_db,
    suggest_journals_by_title,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(page_title="OpenSIGAPS", page_icon="🧬", layout="wide")

st.markdown(
    """
<style>
/* ── FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── DESIGN TOKENS ── */
:root {
    --bg-app:         #f4f7fb;
    --bg-surface:     #ffffff;
    --bg-elevated:    #f8fafd;
    --bg-hover:       #edf2fb;
    --navy:           #1e3a5f;
    --blue:           #2563eb;
    --blue-light:     #3b82f6;
    --blue-dim:       rgba(37,99,235,0.08);
    --blue-border:    rgba(37,99,235,0.20);
    --gold:           #c8921a;
    --gold-dim:       rgba(200,146,26,0.09);
    --gold-border:    rgba(200,146,26,0.22);
    --text-primary:   #0f1e35;
    --text-secondary: #2d4163;
    --text-muted:     #6b7fa3;
    --text-faint:     #a8b8d8;
    --border-subtle:  #dde6f5;
    --border-card:    #c8d8ee;
    --success:        #0f766e;
    --success-dim:    rgba(15,118,110,0.07);
    --success-border: rgba(15,118,110,0.22);
    --warning:        #b45309;
    --warning-dim:    rgba(180,83,9,0.07);
    --warning-border: rgba(180,83,9,0.20);
    --danger:         #be123c;
    --danger-dim:     rgba(190,18,60,0.07);
    --danger-border:  rgba(190,18,60,0.18);
    --radius-sm:      8px;
    --radius:         12px;
    --radius-lg:      18px;
    --shadow-sm:      0 1px 4px rgba(30,58,95,0.07), 0 1px 2px rgba(30,58,95,0.04);
    --shadow:         0 4px 16px rgba(30,58,95,0.09), 0 2px 6px rgba(30,58,95,0.05);
    --shadow-lg:      0 12px 40px rgba(30,58,95,0.12), 0 4px 12px rgba(30,58,95,0.06);
    --shadow-blue:    0 4px 20px rgba(37,99,235,0.20);
    --font-display:   'Playfair Display', Georgia, serif;
    --font-body:      'Sora', system-ui, sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
}

/* ── MASQUER LE CHROME STREAMLIT ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── FOND PRINCIPAL ── */
[data-testid="stAppViewContainer"], .main {
    background: var(--bg-app) !important;
}
[data-testid="stAppViewContainer"]::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(ellipse 70% 50% at 5% 0%, rgba(37,99,235,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 45% at 95% 100%, rgba(200,146,26,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(30,58,95,0.02) 0%, transparent 70%);
}
.block-container {
    background: transparent !important;
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 1400px !important;
    position: relative; z-index: 1;
    font-family: var(--font-body) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none !important;
}
[data-testid="stSidebar"]::before {
    content: ''; display: block; height: 3px;
    background: linear-gradient(90deg, var(--blue-light) 0%, var(--gold) 100%);
}
[data-testid="stSidebarContent"] { padding: 1.6rem 1.3rem !important; }
[data-testid="stSidebar"] h1 {
    font-family: var(--font-display) !important; font-size: 1.45rem !important;
    font-weight: 600 !important; color: #ffffff !important;
    letter-spacing: -0.01em !important; margin-bottom: 0.15rem !important;
}
[data-testid="stSidebar"] .stCaption p {
    color: rgba(255,255,255,0.45) !important; font-family: var(--font-mono) !important;
    font-size: 0.64rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(255,255,255,0.75) !important; font-size: 0.82rem !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; margin: 1.2rem 0 !important; }
[data-testid="stSidebar"] .stNumberInput > div > div > input,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.18) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] p { color: rgba(255,255,255,0.5) !important; }

/* ── TITRES ── */
h1 {
    font-family: var(--font-display) !important; font-size: 2rem !important;
    font-weight: 700 !important; color: var(--navy) !important;
    letter-spacing: -0.02em !important; line-height: 1.2 !important;
}
h2 {
    font-family: var(--font-display) !important; font-size: 1.25rem !important;
    font-weight: 600 !important; color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding-bottom: 0.45rem !important; margin-top: 1.8rem !important;
}
h3 {
    font-family: var(--font-body) !important; font-size: 0.68rem !important;
    font-weight: 700 !important; color: var(--text-muted) !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important;
}
p, li { font-family: var(--font-body) !important; color: var(--text-secondary) !important; line-height: 1.75 !important; font-size: 0.88rem !important; }
.stCaption p { color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* ── ONGLETS — pill design ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important; border-radius: var(--radius) !important;
    padding: 5px !important; gap: 3px !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-sm) !important; width: fit-content !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important; padding: 9px 22px !important;
    font-family: var(--font-body) !important; font-size: 0.84rem !important;
    font-weight: 500 !important; border: none !important; letter-spacing: 0.01em !important;
    transition: all 0.18s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--navy) !important; background: var(--bg-hover) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%) !important;
    color: #ffffff !important; font-weight: 600 !important;
    box-shadow: var(--shadow-blue) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── BOUTONS ── */
.stButton > button {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important; font-weight: 600 !important;
    font-size: 0.83rem !important; letter-spacing: 0.04em !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.22) !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--blue) 0%, var(--blue-light) 100%) !important;
    box-shadow: var(--shadow-blue) !important; transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button[disabled] {
    background: var(--border-subtle) !important; color: var(--text-faint) !important;
    box-shadow: none !important; cursor: not-allowed !important; transform: none !important;
}

/* Download button — ghost style ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1.5px solid var(--blue) !important; color: var(--blue) !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--blue-dim) !important; transform: translateY(-1px) !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.15) !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-surface) !important; border: 1.5px solid var(--border-card) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
    font-family: var(--font-body) !important; font-size: 0.89rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--blue) !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: var(--text-faint) !important; }
[data-baseweb="select"] > div {
    background: var(--bg-surface) !important; border: 1.5px solid var(--border-card) !important;
    color: var(--text-primary) !important; box-shadow: var(--shadow-sm) !important;
    border-radius: var(--radius-sm) !important;
}
[data-baseweb="menu"] {
    background: var(--bg-surface) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important; box-shadow: var(--shadow-lg) !important;
}
[data-baseweb="option"] { background: transparent !important; color: var(--text-secondary) !important; font-family: var(--font-body) !important; font-size: 0.87rem !important; }
[data-baseweb="option"]:hover { background: var(--bg-hover) !important; color: var(--navy) !important; }
.stNumberInput button { background: var(--bg-elevated) !important; border-color: var(--border-subtle) !important; color: var(--text-muted) !important; }

/* Labels */
.stSelectbox label, .stTextInput label, .stNumberInput label,
.stTextArea label, .stSlider label {
    font-family: var(--font-body) !important; font-size: 0.72rem !important;
    font-weight: 700 !important; color: var(--text-muted) !important;
    text-transform: uppercase !important; letter-spacing: 0.09em !important;
}

/* ── MÉTRIQUES ── */
[data-testid="metric-container"] {
    background: var(--bg-surface) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important; padding: 1.3rem 1.5rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease !important;
    position: relative; overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--navy) 0%, var(--blue) 60%, var(--blue-light) 100%);
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important; box-shadow: var(--shadow) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-family: var(--font-body) !important; font-size: 0.68rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.11em !important; color: var(--text-muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important; font-size: 1.8rem !important;
    font-weight: 600 !important; color: var(--navy) !important; line-height: 1.2 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--font-body) !important; font-size: 0.72rem !important;
}

/* ── DATAFRAME / EDITOR — laisser config.toml gérer les couleurs du canvas ── */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
}
.dvn-scroller { overflow: auto !important; }

/* ── ALERTS ── */
[data-testid="stSuccess"] { background: var(--success-dim) !important; border: 1px solid var(--success-border) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stError"]   { background: var(--danger-dim)  !important; border: 1px solid var(--danger-border)  !important; border-radius: var(--radius-sm) !important; }
[data-testid="stInfo"]    { background: var(--blue-dim)    !important; border: 1px solid var(--blue-border)    !important; border-radius: var(--radius-sm) !important; }
[data-testid="stWarning"] { background: var(--warning-dim) !important; border: 1px solid var(--warning-border) !important; border-radius: var(--radius-sm) !important; }

/* ── EXPANDERS — fix arrow overlap bug ── */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] details > summary {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    padding: 12px 18px !important;
    cursor: pointer !important;
    background: var(--bg-elevated) !important;
    border-radius: var(--radius-sm) !important;
    gap: 10px !important;
    list-style: none !important;
    user-select: none !important;
}
[data-testid="stExpander"] details[open] > summary {
    background: var(--bg-hover) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}
[data-testid="stExpander"] details > summary:hover { background: var(--bg-hover) !important; }
/* Cacher le marqueur natif du details et le texte parasite */
[data-testid="stExpander"] details > summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] details > summary::marker { display: none !important; }
/* L'icône toggle de Streamlit */
[data-testid="stExpanderToggleIcon"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 20px !important; height: 20px !important;
    flex-shrink: 0 !important;
    color: var(--blue) !important;
    font-size: 0 !important;           /* cache le texte _arrow si présent */
    line-height: 0 !important;
}
[data-testid="stExpanderToggleIcon"] svg {
    width: 16px !important; height: 16px !important;
    fill: var(--blue) !important; color: var(--blue) !important;
    font-size: initial !important;     /* SVG hérite normalement */
}
[data-testid="stExpander"] details > summary > div,
[data-testid="stExpander"] details > summary > div > p {
    font-family: var(--font-body) !important; font-size: 0.86rem !important;
    font-weight: 500 !important; color: var(--text-secondary) !important;
    margin: 0 !important; flex: 1 !important;
}
[data-testid="stExpander"] details > summary:hover > div > p { color: var(--navy) !important; }

/* ── DIVIDERS ── */
hr, [data-testid="stDivider"] { border-color: var(--border-subtle) !important; margin: 1.8rem 0 !important; }

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div { background: var(--border-subtle) !important; border-radius: 99px !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--navy), var(--blue)) !important; border-radius: 99px !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-app); }
::-webkit-scrollbar-thumb { background: var(--border-card); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

/* ── COMPOSANTS CUSTOM ── */
.warn-banner { background: var(--warning-dim); border: 1px solid var(--warning-border); border-left: 3px solid var(--warning); border-radius: var(--radius-sm); padding: 12px 18px; font-family: var(--font-body); font-size: 0.83rem; color: var(--text-secondary); line-height: 1.7; }
.info-banner { background: var(--blue-dim); border: 1px solid var(--blue-border); border-left: 3px solid var(--blue); border-radius: var(--radius-sm); padding: 14px 18px; font-family: var(--font-body); font-size: 0.83rem; color: var(--text-secondary); line-height: 1.7; }
.rule-box { background: var(--gold-dim); border: 1px solid var(--gold-border); border-left: 3px solid var(--gold); border-radius: var(--radius-sm); padding: 14px 18px; font-family: var(--font-body); font-size: 0.83rem; color: var(--text-secondary); line-height: 1.7; margin: 10px 0; }
.rule-box b, .info-banner b, .warn-banner b { color: var(--text-primary); font-weight: 600; }
.rule-box a, .info-banner a, .warn-banner a { color: var(--blue); text-decoration: none; border-bottom: 1px solid var(--blue-border); }

/* ── TABLEAUX MARKDOWN ── */
table { width: 100%; border-collapse: collapse; font-family: var(--font-body) !important; font-size: 0.85rem !important; }
thead tr { background: #eef3fb !important; border-bottom: 2px solid var(--blue-border) !important; }
thead th { color: var(--navy) !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; font-size: 0.7rem !important; padding: 10px 14px !important; }
tbody tr { border-bottom: 1px solid var(--border-subtle) !important; transition: background 0.12s !important; }
tbody tr:hover { background: var(--bg-hover) !important; }
tbody td { color: var(--text-secondary) !important; padding: 9px 14px !important; }

/* ── CODE ── */
code, pre { font-family: var(--font-mono) !important; background: var(--bg-elevated) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-sm) !important; color: var(--navy) !important; font-size: 0.82rem !important; }

/* ── CHECKBOX / RADIO ── */
input[type="checkbox"] { accent-color: var(--blue) !important; }
[data-testid="stCheckbox"] label { text-transform: none !important; letter-spacing: 0 !important; font-size: 0.84rem !important; color: var(--text-secondary) !important; }
[data-testid="stRadio"] > div { gap: 6px !important; }
[data-testid="stRadio"] label {
    background: var(--bg-surface) !important; border: 1.5px solid var(--border-card) !important;
    border-radius: var(--radius-sm) !important; padding: 10px 22px !important;
    font-size: 0.9rem !important; font-weight: 500 !important; color: var(--text-secondary) !important;
    cursor: pointer !important; transition: all 0.18s ease !important; box-shadow: var(--shadow-sm) !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%) !important;
    border-color: var(--blue) !important; color: #ffffff !important;
    font-weight: 600 !important; box-shadow: var(--shadow-blue) !important;
}

/* ── MULTISELECT TAGS ── */
[data-baseweb="tag"] { background: #dbeafe !important; border-color: #93c5fd !important; color: #1d4ed8 !important; }
[data-baseweb="tag"] span { color: #1d4ed8 !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] > div > div > div > div { background: var(--blue) !important; }


/* ── SIDEBAR : labels et slider plus visibles ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.88) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider p {
    color: rgba(255,255,255,0.75) !important;
}
/* Help icon en sidebar → fond transparent */
[data-testid="stSidebar"] button[data-testid="baseButton-header"],
[data-testid="stSidebar"] button[title="Help"],
[data-testid="stSidebar"] .stTooltipIcon > button,
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] > button {
    background: transparent !important;
    border: none !important;
    color: rgba(255,255,255,0.55) !important;
}

/* ── Onglets principaux : texte blanc sur fond navy ── */
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
}
.stTabs [aria-selected="false"] {
    color: var(--text-secondary) !important;
}

/* ── Sous-onglets "J'ai une idée" / "Aide" : texte blanc ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] [aria-selected="true"] {
    color: #ffffff !important;
}


/* ── SIDEBAR — labels homogènes (tous composants) ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.88) !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
/* Slider – valeurs des années lisibles */
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider p,
[data-testid="stSidebar"] .stSlider span { color: rgba(255,255,255,0.80) !important; }

/* ── HELP ICON — fond TOTALEMENT transparent dans la sidebar ──
   Streamlit rend le tooltip via différents composants selon la version.
   On cible tous les cas possibles : conteneur, bouton, SVG, pseudo-éléments. */
[data-testid="stSidebar"] [data-testid="stTooltipIcon"],
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] ~ div button,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] ~ div button *,
[data-testid="stSidebar"] button[data-testid="baseButton-header"],
[data-testid="stSidebar"] button[data-testid="baseButton-header"] *,
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebar"] button[kind="header"] * {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
/* L'icône SVG elle-même : visible mais fondue dans le fond navy */
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] ~ div button svg,
[data-testid="stSidebar"] button[data-testid="baseButton-header"] svg,
[data-testid="stSidebar"] button[kind="header"] svg {
    color: rgba(255,255,255,0.42) !important;
    fill: rgba(255,255,255,0.42) !important;
    stroke: none !important;
}

/* ── ONGLETS actifs — texte blanc ── */
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] * { color: #ffffff !important; }

/* ── BOUTONS principaux — texte blanc ── */
.stButton > button,
.stButton > button * { color: #ffffff !important; }
[data-testid="stDownloadButton"] > button,
[data-testid="stDownloadButton"] > button * { color: var(--blue) !important; }

/* ── FILTRES COMPACTS : override radio pour zone tri ── */
.sort-controls [data-testid="stRadio"] label {
    padding: 4px 12px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
.sort-controls [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 34px !important;
    font-size: 0.82rem !important;
}


/* ── ANIMATIONS ── */
@keyframes fadeSlideUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes shimmer { from { background-position: -200% 0; } to { background-position: 200% 0; } }

.block-container > div > div > div { animation: fadeSlideUp 0.38s cubic-bezier(0.4,0,0.2,1) both; }
.block-container > div > div > div:nth-child(1) { animation-delay: 0.00s; }
.block-container > div > div > div:nth-child(2) { animation-delay: 0.06s; }
.block-container > div > div > div:nth-child(3) { animation-delay: 0.12s; }
.block-container > div > div > div:nth-child(4) { animation-delay: 0.18s; }
.block-container > div > div > div:nth-child(5) { animation-delay: 0.24s; }

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR — labels, help icon, sliders
════════════════════════════════════════════════════════════════ */
/* Labels (toutes les variantes Streamlit) */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label {
    color: rgba(255,255,255,0.88) !important;
    font-family: var(--font-body) !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* Help icon — sélecteur universel : TOUS les boutons de la sidebar
   sauf les steppers +/- du number_input (identifiés par aria-label) */
[data-testid="stSidebar"] button {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
/* Steppers +/- : rétablir un fond minimal pour qu'ils soient visibles */
[data-testid="stSidebar"] button[aria-label="Increment"],
[data-testid="stSidebar"] button[aria-label="Decrement"],
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: #ffffff !important;
}
/* SVG du help icon : couleur discrète */
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg path,
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg circle {
    color: rgba(255,255,255,0.50) !important;
    fill: rgba(255,255,255,0.50) !important;
    stroke: rgba(255,255,255,0.50) !important;
}
/* Slider ticks & valeurs */
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider p,
[data-testid="stSidebar"] .stSlider span { color: rgba(255,255,255,0.80) !important; }

/* ═══════════════════════════════════════════════════════════════
   ONGLETS — texte actif blanc
════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span,
.stTabs [data-baseweb="tab"][aria-selected="true"] div,
.stTabs [role="tab"][aria-selected="true"],
.stTabs [role="tab"][aria-selected="true"] * {
    color: #ffffff !important;
}

/* ═══════════════════════════════════════════════════════════════
   BOUTONS — texte blanc (gradient navy→blue)
════════════════════════════════════════════════════════════════ */
div.stButton > button,
div.stButton > button p,
div.stButton > button span,
div.stButton > button div,
div.stButton > button * { color: #ffffff !important; }
/* Download button ghost (texte bleu) */
div[data-testid="stDownloadButton"] > button,
div[data-testid="stDownloadButton"] > button * { color: var(--blue) !important; }

/* ═══════════════════════════════════════════════════════════════
   ZONE TRI/FILTRES — compacte
════════════════════════════════════════════════════════════════ */
/* Toggle compact */
[data-testid="stToggle"] { transform: scale(0.82); transform-origin: left center; }
/* Selectbox réduit */
.filter-row [data-baseweb="select"] > div { min-height: 34px !important; font-size: 0.82rem !important; }
/* Labels filter invisibles */
.filter-row label { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

ALL_POSITIONS = ["1er", "2ème", "3ème", "ADA", "Dernier", "Autre"]
ALL_RANKS = ["A+", "A", "B", "C", "D", "E", "NC"]

# ─────────────────────────────────────────────
# MODÈLE D'EMBEDDING — ONNX > PyTorch > Jaccard
# ─────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_embed_model():
    """Délègue au sélecteur de backend dans backend.py. Retourne (model, backend_str)."""
    return load_embed_model()


# ─────────────────────────────────────────────
# SIGAPS REFERENCE DATABASE — chargement CSV
# @st.cache_resource (pas cache_data) : on veut conserver la matrice numpy
# en mémoire sans copie. cache_data sérialiserait l'objet et perdrait la matrice.
# ─────────────────────────────────────────────


@st.cache_resource(show_spinner="Chargement du référentiel SIGAPS…")
def _load_ref_db(csv_path: str) -> SigapsRefDB:
    return SigapsRefDB().load(csv_path)


# Chemin par défaut : même dossier que app.py
_CSV_PATH = str(Path(__file__).parent / "sigaps_ref.csv")

# ── Chargement du modèle
_embed_model, _embed_backend = _load_embed_model()
set_embed_model(_embed_model, _embed_backend)

if Path(_CSV_PATH).exists():
    _ref_db = _load_ref_db(_CSV_PATH)
    set_ref_db(_ref_db)
    _CSV_STATUS = (True, len(_ref_db), _ref_db.source_path)

    # ── Construction des embeddings si le modèle est disponible ──────────────
    # Lancé après le chargement CSV, une seule fois (cache disque ensuite).
    if _embed_model is not None and not _ref_db.embeddings_ready:
        _emb_placeholder = st.empty()
        _emb_bar = _emb_placeholder.progress(
            0, text="🧠 Calcul des embeddings journaux (1ʳᵉ exécution)…"
        )

        def _emb_progress(pct: int):
            _emb_bar.progress(pct, text=f"🧠 Embeddings journaux… {pct}%")

        _ref_db.build_embeddings(_embed_model, progress_callback=_emb_progress)
        _emb_placeholder.empty()
else:
    _ref_db = SigapsRefDB()
    _CSV_STATUS = (False, 0, "")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "articles_data" not in st.session_state:
    st.session_state.articles_data: list[Article] = []
if "team_results" not in st.session_state:
    st.session_state.team_results: dict = {}
if "team_member_arts" not in st.session_state:
    st.session_state.team_member_arts: dict[str, list[Article]] = {}
if "raw_suggestions" not in st.session_state:
    st.session_state.raw_suggestions: list = []
if "suggest_query_cache" not in st.session_state:
    st.session_state.suggest_query_cache: str = ""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ OpenSIGAPS")
    st.caption("SIGAPS · MERRI 2022 · PubMed")
    st.markdown("---")

    valeur_point = st.number_input(
        "Valeur du Point SIGAPS (€)",
        value=VALEUR_POINT_EUROS,
        step=10,
        help="Valeur annuelle fixée par votre établissement.",
    )

    institution_filter = None

    current_year = datetime.now().year
    year_range = st.slider(
        "Fenêtre temporelle",
        min_value=current_year - 15,
        max_value=current_year,
        value=(current_year - 4, current_year),
        help="SIGAPS calcule la dotation sur 4 ans glissants.",
    )

    st.markdown("---")

    # ── Statut du référentiel CSV ──
    _csv_ok, _csv_n, _csv_path_display = _CSV_STATUS
    if _csv_ok:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.22);"
            f"border-radius:8px;padding:10px 14px;"
            f"font-family:Sora,sans-serif;font-size:0.8rem;color:rgba(255,255,255,0.7);'>"
            f"<span style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;'>Référentiel SIGAPS</span><br>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.05rem;font-weight:700;color:#ffffff;'>"
            f"{_csv_n:,} journaux</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='warn-banner'>⚠️ <b>sigaps_ref.csv non trouvé.</b><br>"
            "Rangs estimés par heuristique — placez le fichier à côté d'app.py.<br>"
            "Vérifier sur <a href='https://www.sigaps.fr'>sigaps.fr</a>.</div>",
            unsafe_allow_html=True,
        )

    # ── Statut moteur NLP ──
    st.markdown("---")
    if _embed_model is None:
        _nb, _nl, _nt = "rgba(217,119,6,0.07)", "#d97706", "⚠️ Moteur : Jaccard"
        _ns = "Installez optimum[onnxruntime] ou sentence-transformers."
    elif _embed_backend == "onnx":
        _nb, _nl, _nt = "rgba(5,150,105,0.07)", "#059669", "⚡ ONNX Runtime"
        _ns = f"multilingual-e5-small · {len(_ref_db):,} journaux indexés"
    else:
        _nb, _nl, _nt = "rgba(37,99,235,0.07)", "#2563eb", "🧠 PyTorch actif"
        _ns = f"multilingual-e5-small · {len(_ref_db):,} journaux indexés"
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.20);"
        f"border-radius:8px;padding:9px 13px;"
        f"font-size:0.76rem;color:rgba(255,255,255,0.8);'>"
        f"<b style='color:{_nl};'>{_nt}</b><br>"
        f"<span style='color:rgba(255,255,255,0.5);font-size:0.72rem;'>{_ns}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────

tab_analyse, tab_journal, tab_methodo = st.tabs(
    ["🧬 Analyse SIGAPS", "🔎 Quel journal choisir ?", "📖 Méthodologie"]
)


# ═══════════════════════════════════════════════════════════════
# ONGLET 1 — ANALYSE SIGAPS (individuel + équipe fusionnés)
# ═══════════════════════════════════════════════════════════════

with tab_analyse:
    # ── Sélecteur de mode ──────────────────────────────────────
    _mode = st.radio(
        "Mode d'analyse",
        options=["👤 Individuel", "👥 Équipe / Service"],
        horizontal=True,
        label_visibility="collapsed",
        key="analyse_mode",
    )

    st.markdown(
        """
    <style>
    div[data-testid='stRadio'] > div { gap: 6px !important; margin-bottom: 1.2rem; }
    div[data-testid='stRadio'] label {
        background: var(--bg-surface) !important;
        border: 1.5px solid var(--border-card) !important;
        border-radius: 8px !important;
        padding: 9px 22px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        cursor: pointer !important;
        transition: all 0.18s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    div[data-testid='stRadio'] label:has(input:checked) {
        background: var(--blue) !important;
        border-color: var(--blue) !important;
        color: #fff !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(37,99,235,0.25) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ════════════════════════════════════════════════
    # MODE INDIVIDUEL
    # ════════════════════════════════════════════════
    if _mode == "👤 Individuel":
        col1, col2 = st.columns([3, 1])
        with col1:
            researcher_name = st.text_input(
                "Nom du chercheur (Prénom Nom)",
                placeholder="Ex: Romain Buono",
            )
        with col2:
            search_btn = st.button("🚀 Lancer l'analyse", use_container_width=True)

        if search_btn and researcher_name:
            search_service = FederatedSearch(ref_db=_ref_db)
            with st.spinner("Interrogation PubMed en cours…"):
                results = search_service.search(
                    researcher_name, institution_filter=institution_filter
                )
                st.session_state.articles_data = results

            if not results:
                st.error("Aucune publication trouvée. Vérifiez l'orthographe.")
            else:
                st.success(f"✅ **{len(results)} articles** récupérés via PubMed")

        if st.session_state.articles_data:
            st.divider()

            data_for_df = []
            for a in st.session_state.articles_data:
                in_range = year_range[0] <= a.publication_year <= year_range[1]
                data_for_df.append(
                    {
                        "is_selected": a.is_selected and in_range,
                        "title": a.title,
                        "journal_name": a.journal_name,
                        "publication_year": a.publication_year,
                        "my_position": a.my_position,
                        "estimated_rank": a.estimated_rank,
                        "rank_source": "Valeur réelle"
                        if getattr(a, "rank_source", "") == "csv"
                        else "〜 Estimé",
                        "nb_authors": a.nb_authors,
                        "id": a.id,
                        "doi": a.doi,
                        "nlm_id": getattr(a, "nlm_unique_id", ""),
                        "authors_str": "; ".join(a.authors_list[:5]),
                    }
                )

            df_source = pd.DataFrame(data_for_df)

            st.subheader("📝 Revue des publications")
            _hcol1, _hcol2 = st.columns([5, 1])
            with _hcol1:
                st.caption(
                    f"✏️ Décochez les homonymes · Corrigez Position et Rang · "
                    f"Fenêtre active : {year_range[0]}–{year_range[1]}"
                )
            with _hcol2:
                _all_sel_indiv = st.checkbox(
                    "Tout sélect.",
                    value=True,
                    key="indiv_select_all",
                    help="Cocher = tout sélectionner · Décocher = tout désélectionner",
                )
                df_source["is_selected"] = _all_sel_indiv

            # Hauteur dynamique : 35 px/ligne + 38 px header, min 200, max 520
            _n_rows = len(df_source)
            _height = min(max(_n_rows * 35 + 38, 200), 520) if _n_rows < 10 else 520

            edited_df = st.data_editor(
                df_source,
                column_config={
                    "is_selected": st.column_config.CheckboxColumn(
                        "✅ Validé", width="small"
                    ),
                    "title": st.column_config.TextColumn(
                        "Titre", width="large", disabled=True
                    ),
                    "journal_name": st.column_config.TextColumn("Revue", disabled=True),
                    "publication_year": st.column_config.NumberColumn(
                        "Année", format="%d", disabled=True
                    ),
                    "my_position": st.column_config.SelectboxColumn(
                        "Position",
                        options=ALL_POSITIONS,
                        required=True,
                        help="ADA = avant-dernier (C1=3), seulement si ≥ 6 auteurs.",
                    ),
                    "estimated_rank": st.column_config.SelectboxColumn(
                        "Rang SIGAPS",
                        options=ALL_RANKS,
                        required=True,
                        help="A+ = 6 grandes revues uniquement (NEJM, Lancet…).",
                    ),
                    "nb_authors": st.column_config.NumberColumn(
                        "Nb auteurs",
                        format="%d",
                        min_value=1,
                        help="Total d'auteurs sur l'article — nécessaire pour le score fractionnaire.",
                    ),
                    "rank_source": st.column_config.TextColumn(
                        "Source rang",
                        width="small",
                        disabled=True,
                        help="Valeur réelle = rang officiel sigaps_ref.csv · 〜 Estimé = heuristique",
                    ),
                    "id": None,
                    "doi": None,
                    "authors_str": None,
                    "nlm_id": None,
                },
                hide_index=True,
                use_container_width=True,
                height=_height,
            )

            # ── Calculs ──
            total_presence = 0.0
            total_frac = 0.0
            validated_rows = []

            for _, row in edited_df.iterrows():
                p_pts = f_pts = 0.0
                if row["is_selected"]:
                    p_pts = calculate_presence_score(
                        row["my_position"], row["estimated_rank"]
                    )
                    nb = int(row.get("nb_authors", 0) or 0)
                    if nb > 0:
                        f_pts = calculate_fractional_score(
                            row["my_position"], row["estimated_rank"], nb
                        )
                total_presence += p_pts
                total_frac += f_pts
                validated_rows.append(
                    {**row.to_dict(), "pts_presence": p_pts, "pts_frac": f_pts}
                )

            annual_presence = total_presence * valeur_point
            annual_frac = total_frac * valeur_point

            st.divider()
            st.subheader("💰 Valorisation Financière Individuelle")

            st.markdown(
                "<div class='info-banner'>"
                "📌 <b>Score Présence</b> (C1×C2) : votre contribution individuelle à la "
                "valorisation de l'établissement — argument pour la négociation de poste HU.<br>"
                "📌 <b>Score Fractionnaire</b> ((C1/ΣC1)×C2) : votre quote-part réelle dans "
                "la dotation MERRI, diminue avec le nombre de co-auteurs."
                "</div>",
                unsafe_allow_html=True,
            )
            st.write("")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Points Présence", f"{total_presence:.1f}")
            c2.metric(
                "Valeur annuelle (score × valeur point)", f"{annual_presence:,.0f} €"
            )
            c3.metric("Points Fractionnaires (MERRI)", f"{total_frac:.2f}")
            c4.metric(
                "Valeur 4 ans",
                f"{annual_presence * 4:,.0f} €",
                delta="Argument négociation",
            )

            nb_validated = sum(1 for r in validated_rows if r["is_selected"])
            st.caption(f"Articles validés : {nb_validated} / {len(validated_rows)}")

            with st.expander("Détail par position / rang"):
                detail = []
                for pos in ALL_POSITIONS:
                    for rank in ALL_RANKS:
                        count = sum(
                            1
                            for r in validated_rows
                            if r["is_selected"]
                            and r["my_position"] == pos
                            and r["estimated_rank"] == rank
                        )
                        if count > 0:
                            p_each = calculate_presence_score(pos, rank)
                            detail.append(
                                {
                                    "Position": pos,
                                    "Rang": rank,
                                    "C1": C1_COEFFICIENTS.get(pos, 1),
                                    "C2": C2_COEFFICIENTS.get(rank, 1),
                                    "Nb articles": count,
                                    "Pts Présence/article": p_each,
                                    "Total Présence": p_each * count,
                                    "Valeur/an (€)": p_each * count * valeur_point,
                                }
                            )
                if detail:
                    st.dataframe(
                        pd.DataFrame(detail), use_container_width=True, hide_index=True
                    )

            st.divider()
            st.subheader("📥 Export")

            export_rows = [
                {
                    "Validé": r["is_selected"],
                    "Titre": r["title"],
                    "Revue": r["journal_name"],
                    "Année": r["publication_year"],
                    "Position": r["my_position"],
                    "Rang": r["estimated_rank"],
                    "Nb auteurs": r.get("nb_authors", 0),
                    "C1": C1_COEFFICIENTS.get(r["my_position"], 1),
                    "C2": C2_COEFFICIENTS.get(r["estimated_rank"], 1),
                    "Pts Présence": r["pts_presence"],
                    "Pts Fractionnaire": round(r["pts_frac"], 4),
                    "NLM_ID": r.get("nlm_id", ""),
                    "Source rang": r.get("rank_source", ""),
                    "DOI": r.get("doi", ""),
                }
                for r in validated_rows
            ]

            summary_rows = [
                {
                    "Indicateur": "Score Présence total",
                    "Valeur": f"{total_presence:.1f} pts",
                },
                {
                    "Indicateur": "Score Fractionnaire total",
                    "Valeur": f"{total_frac:.2f} pts",
                },
                {
                    "Indicateur": "Valeur annuelle (score × valeur point)",
                    "Valeur": f"{annual_presence:,.0f} €",
                },
                {
                    "Indicateur": "Valeur 4 ans",
                    "Valeur": f"{annual_presence * 4:,.0f} €",
                },
                {"Indicateur": "Valeur du point", "Valeur": f"{valeur_point} €"},
                {
                    "Indicateur": "Fenêtre temporelle",
                    "Valeur": f"{year_range[0]}–{year_range[1]}",
                },
                {"Indicateur": "Algorithme", "Valeur": "MERRI 2022"},
            ]

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                pd.DataFrame(export_rows).to_excel(
                    writer, sheet_name="Publications", index=False
                )
                pd.DataFrame(summary_rows).to_excel(
                    writer, sheet_name="Valorisation", index=False
                )

            name_safe = (
                researcher_name.replace(" ", "_") if researcher_name else "export"
            )
            st.download_button(
                "⬇️ Télécharger le rapport Excel",
                data=buffer.getvalue(),
                file_name=f"SIGAPS_{name_safe}_{datetime.now().strftime('%Y%m')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ═══════════════════════════════════════════════════════════════
    # ONGLET 2 — ÉQUIPE / SERVICE
    # ═══════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════
    # MODE ÉQUIPE / SERVICE
    # ════════════════════════════════════════════════
    else:
        st.title("👥 Score SIGAPS — Équipe / Service")
        st.markdown(
            "<div class='rule-box'>"
            "📋 <b>Règle officielle SIGAPS (Guide pratique AP-HP) :</b> "
            "Seule la <b>meilleure position</b> est retenue par article (règle service). "
            "Pour le <b>score MERRI</b> : Σ (C1_membre / Σ C1_article) × C2."
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Saisie des membres ──
        st.subheader("1. Composition de l'équipe")
        st.caption(
            "Saisissez les noms des membres de votre équipe (un par ligne). "
            "Chaque chercheur sera interrogé individuellement, puis les résultats seront consolidés."
        )

        default_team = st.session_state.get("team_names_input", "")
        team_names_raw = st.text_area(
            "Membres de l'équipe (Prénom Nom, un par ligne)",
            value=default_team,
            height=150,
            placeholder="Romain Buono\nJean Dupont\nMarie Martin",
        )
        st.session_state["team_names_input"] = team_names_raw

        team_names = [n.strip() for n in team_names_raw.splitlines() if n.strip()]

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            team_search_btn = st.button(
                f"🚀 Analyser l'équipe ({len(team_names)} membre{'s' if len(team_names) != 1 else ''})",
                use_container_width=True,
                disabled=(len(team_names) == 0),
            )
        with col_btn2:
            if st.button("🗑️ Réinitialiser", use_container_width=True):
                st.session_state.team_results = {}
                st.session_state.team_member_arts = {}
                st.rerun()

        if team_search_btn and team_names:
            search_service = FederatedSearch(ref_db=_ref_db)
            member_articles: dict[str, list[Article]] = {}

            progress_bar = st.progress(0, text="Initialisation…")
            for i, name in enumerate(team_names):
                progress_bar.progress(
                    (i) / len(team_names),
                    text=f"Recherche : {name} ({i + 1}/{len(team_names)})…",
                )
                results = search_service.search(
                    name, institution_filter=institution_filter
                )
                # Filtre fenêtre temporelle et sélection
                filtered = [
                    a
                    for a in results
                    if year_range[0] <= a.publication_year <= year_range[1]
                ]
                member_articles[name] = filtered

            progress_bar.progress(1.0, text="Consolidation des résultats…")

            team_score = calculate_team_presence_score(member_articles)
            st.session_state.team_results = team_score
            st.session_state.team_member_arts = member_articles
            progress_bar.empty()
            st.success(
                f"✅ Équipe analysée : **{len(team_names)} membres** · "
                f"**{len(team_score['articles_uniques'])} articles uniques** sur "
                f"{year_range[0]}–{year_range[1]}"
            )

        # ── Résultats équipe ──
        if st.session_state.team_results and st.session_state.team_member_arts:
            team_score = st.session_state.team_results
            member_arts = st.session_state.team_member_arts

            st.divider()

            # ── 2. Tableau éditeur — revue des publications équipe ──
            st.subheader("2. Revue des publications de l'équipe")
            _eq_hcol1, _eq_hcol2 = st.columns([5, 1])
            with _eq_hcol1:
                st.caption(
                    "✏️ Décochez les articles hors-périmètre (homonymes, collaborations externes…) · "
                    "Corrigez la Position ou le Rang si nécessaire · "
                    "Les métriques se recalculent instantanément."
                )
            with _eq_hcol2:
                _all_sel_equipe = st.checkbox(
                    "Tout sélect.",
                    value=True,
                    key="equipe_select_all",
                    help="Cocher = tout sélectionner · Décocher = tout désélectionner",
                )
            st.markdown(
                '<div style="background:rgba(201,168,76,0.13);border:1px solid rgba(201,168,76,0.35);'
                "border-radius:6px;padding:8px 14px;font-family:DM Sans,sans-serif;"
                'font-size:0.82rem;color:#dde4f0;display:inline-block;margin-bottom:6px;">'
                '⌨️ Après chaque modification dans le tableau : <b style="color:#e8cc7a;">Ctrl + Enter</b>'
                " pour valider et recalculer les scores."
                "</div>",
                unsafe_allow_html=True,
            )

            # Construction du DataFrame éditeur à partir des articles uniques
            editor_rows = []
            for a in team_score["articles_uniques"]:
                co_label = (
                    ", ".join(a["co_signataires_equipe"])
                    if a["nb_membres_coauteurs"] > 1
                    else a["best_member"]
                )
                editor_rows.append(
                    {
                        "is_selected": True,
                        "title": a["title"],
                        "journal": a["journal"],
                        "year": a["year"],
                        "best_position": a["best_position"],
                        "rank": a["rank"],
                        "rank_source": "Valeur réelle"
                        if a.get("rank_source") == "csv"
                        else "〜 Estimé",
                        "best_member": a["best_member"],
                        "co_auteurs": co_label,
                        "nb_membres": a["nb_membres_coauteurs"],
                        # Champs cachés pour le recalcul
                        "_doi": a["doi"],
                        "_pts_frac_raw": a["pts_frac"],
                    }
                )

            team_editor_df = pd.DataFrame(editor_rows)
            # Appliquer la checkbox master
            team_editor_df["is_selected"] = _all_sel_equipe

            edited_team_df = st.data_editor(
                team_editor_df,
                column_config={
                    "is_selected": st.column_config.CheckboxColumn(
                        "✅ Validé",
                        width="small",
                    ),
                    "title": st.column_config.TextColumn(
                        "Titre",
                        width="large",
                        disabled=True,
                    ),
                    "journal": st.column_config.TextColumn(
                        "Revue",
                        disabled=True,
                    ),
                    "year": st.column_config.NumberColumn(
                        "Année",
                        format="%d",
                        disabled=True,
                    ),
                    "best_position": st.column_config.SelectboxColumn(
                        "Meilleure position",
                        options=ALL_POSITIONS,
                        required=True,
                        help="Position du membre le mieux placé dans l'équipe pour cet article.",
                    ),
                    "rank": st.column_config.SelectboxColumn(
                        "Rang SIGAPS",
                        options=ALL_RANKS,
                        required=True,
                        help="A+ = 6 grandes revues uniquement. Vérifier sur sigaps.fr.",
                    ),
                    "rank_source": st.column_config.TextColumn(
                        "Source rang",
                        width="small",
                        disabled=True,
                        help="Valeur réelle = rang officiel sigaps_ref.csv · 〜 Estimé = heuristique",
                    ),
                    "best_member": st.column_config.TextColumn(
                        "Auteur retenu",
                        disabled=True,
                        help="Membre de l'équipe en meilleure position.",
                    ),
                    "co_auteurs": st.column_config.TextColumn(
                        "Co-auteurs (équipe)",
                        disabled=True,
                    ),
                    "nb_membres": st.column_config.NumberColumn(
                        "Nb membres",
                        format="%d",
                        disabled=True,
                        help="Nombre de membres de l'équipe co-auteurs de cet article.",
                    ),
                    # Colonnes cachées
                    "_doi": None,
                    "_pts_frac_raw": None,
                    # rank_source est visible, pas caché
                },
                hide_index=True,
                use_container_width=True,
                height=480,
            )

            # ── 3. Recalcul live depuis l'éditeur ──
            total_presence_team = 0.0
            total_frac_team = 0.0
            validated_team_rows = []

            for _, row in edited_team_df.iterrows():
                p_pts = f_pts = 0.0
                if row["is_selected"]:
                    # Score présence équipe : meilleure position × C2 (règle service)
                    p_pts = calculate_presence_score(row["best_position"], row["rank"])
                    # Score fractionnaire : on réutilise la somme pré-calculée des membres
                    # (déjà pondérée par le Σ C1 global de l'article en backend)
                    f_pts = float(row["_pts_frac_raw"])
                total_presence_team += p_pts
                total_frac_team += f_pts
                validated_team_rows.append(
                    {
                        **row.to_dict(),
                        "pts_presence_calc": p_pts,
                        "pts_frac_calc": f_pts,
                    }
                )

            annual_team = total_presence_team * valeur_point
            nb_selected = sum(1 for r in validated_team_rows if r["is_selected"])

            # ── Métriques globales équipe ──
            st.divider()
            st.subheader("3. Score global de l'équipe")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Pts Présence équipe",
                f"{total_presence_team:.1f}",
                help="Règle meilleure position : max C1 × C2 par article validé.",
            )
            col2.metric("Valeur annuelle équipe", f"{annual_team:,.0f} €")
            col3.metric(
                "Pts Fractionnaires équipe",
                f"{total_frac_team:.2f}",
                help="Somme des quotes-parts MERRI de tous les membres sur les articles validés.",
            )
            col4.metric(
                "Valeur 4 ans équipe",
                f"{annual_team * 4:,.0f} €",
                delta="Argument collectif",
            )
            st.caption(f"Articles validés : {nb_selected} / {len(validated_team_rows)}")

            # ── Détail par position/rang ──
            with st.expander("Répartition par position / rang"):
                detail_team = []
                for pos in ALL_POSITIONS:
                    for rank in ALL_RANKS:
                        count = sum(
                            1
                            for r in validated_team_rows
                            if r["is_selected"]
                            and r["best_position"] == pos
                            and r["rank"] == rank
                        )
                        if count > 0:
                            p_each = calculate_presence_score(pos, rank)
                            detail_team.append(
                                {
                                    "Position": pos,
                                    "Rang": rank,
                                    "C1": C1_COEFFICIENTS.get(pos, 1),
                                    "C2": C2_COEFFICIENTS.get(rank, 1),
                                    "Nb articles": count,
                                    "Pts Présence/article": p_each,
                                    "Total Présence": p_each * count,
                                    "Valeur/an (€)": int(p_each * count * valeur_point),
                                }
                            )
                if detail_team:
                    st.dataframe(
                        pd.DataFrame(detail_team),
                        use_container_width=True,
                        hide_index=True,
                    )

            # ── 4. Contributions individuelles (recalculées sur articles validés) ──
            st.divider()
            st.subheader("4. Contributions individuelles")
            st.caption(
                "Score présence de chaque membre sur les articles validés ci-dessus "
                "(C1×C2 individuel — sans règle meilleure position). "
                "La somme peut dépasser le score équipe car les co-signatures sont "
                "comptées une fois par membre ici."
            )

            # Index des titres validés pour filtrer les articles de chaque membre
            validated_titles: set[str] = {
                r["title"] for r in validated_team_rows if r["is_selected"]
            }
            validated_dois: set[str] = {
                r["_doi"] for r in validated_team_rows if r["is_selected"] and r["_doi"]
            }

            indiv_data = []
            for name, arts in member_arts.items():
                # Un article membre est "validé" si son titre ou DOI est dans les validés
                member_valid = [
                    a
                    for a in arts
                    if (a.doi and a.doi in validated_dois)
                    or a.title in validated_titles
                ]
                pts_p = sum(
                    calculate_presence_score(a.my_position, a.estimated_rank)
                    for a in member_valid
                )
                pts_f = sum(
                    calculate_fractional_score(
                        a.my_position, a.estimated_rank, a.nb_authors
                    )
                    for a in member_valid
                    if a.nb_authors > 0
                )
                nb_co = sum(
                    1
                    for r in validated_team_rows
                    if r["is_selected"]
                    and r["nb_membres"] > 1
                    and name in r["co_auteurs"]
                )
                indiv_data.append(
                    {
                        "Chercheur": name,
                        "Articles validés": len(member_valid),
                        "Dont co-signés équipe": nb_co,
                        "Pts Présence indiv.": round(pts_p, 1),
                        "Valeur annuelle (€)": int(pts_p * valeur_point),
                        "Pts Frac. indiv.": round(pts_f, 2),
                    }
                )

            st.dataframe(
                pd.DataFrame(indiv_data), use_container_width=True, hide_index=True
            )

            # ── Articles co-signés (filtrés sur validés) ──
            with st.expander("Articles co-signés par plusieurs membres"):
                co_signed = [
                    r
                    for r in validated_team_rows
                    if r["is_selected"] and r["nb_membres"] > 1
                ]
                if co_signed:
                    st.caption(
                        f"{len(co_signed)} article(s) co-signés par ≥2 membres. "
                        "Seule la meilleure position est retenue pour le score équipe (règle service)."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Titre": r["title"][:80] + "…"
                                    if len(r["title"]) > 80
                                    else r["title"],
                                    "Revue": r["journal"],
                                    "Année": r["year"],
                                    "Rang": r["rank"],
                                    "Co-auteurs (équipe)": r["co_auteurs"],
                                    "Meilleure position": r["best_position"],
                                    "Auteur retenu": r["best_member"],
                                    "Pts (équipe)": r["pts_presence_calc"],
                                }
                                for r in co_signed
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Aucun article co-signé validé dans la sélection actuelle.")

            # ── Export équipe ──
            st.divider()
            st.subheader("📥 Export équipe")

            export_team_rows = [
                {
                    "Validé": r["is_selected"],
                    "Titre": r["title"],
                    "Revue": r["journal"],
                    "Année": r["year"],
                    "Auteur retenu": r["best_member"],
                    "Position retenue": r["best_position"],
                    "Rang SIGAPS": r["rank"],
                    "C1": C1_COEFFICIENTS.get(r["best_position"], 1),
                    "C2": C2_COEFFICIENTS.get(r["rank"], 1),
                    "Nb membres coauteurs": r["nb_membres"],
                    "Co-auteurs équipe": r["co_auteurs"],
                    "Pts Présence": r["pts_presence_calc"],
                    "Pts Fractionnaire": round(r["pts_frac_calc"], 4),
                    "Valeur/an (€)": int(r["pts_presence_calc"] * valeur_point),
                    "DOI": r["_doi"],
                }
                for r in validated_team_rows
            ]

            team_summary = [
                {
                    "Indicateur": "Membres de l'équipe",
                    "Valeur": ", ".join(member_arts.keys()),
                },
                {
                    "Indicateur": "Articles validés équipe",
                    "Valeur": f"{nb_selected} / {len(validated_team_rows)}",
                },
                {
                    "Indicateur": "Pts Présence équipe",
                    "Valeur": f"{total_presence_team:.1f}",
                },
                {
                    "Indicateur": "Pts Fractionnaire équipe",
                    "Valeur": f"{total_frac_team:.2f}",
                },
                {
                    "Indicateur": "Valeur annuelle équipe",
                    "Valeur": f"{annual_team:,.0f} €",
                },
                {
                    "Indicateur": "Valeur 4 ans équipe",
                    "Valeur": f"{annual_team * 4:,.0f} €",
                },
                {"Indicateur": "Valeur du point", "Valeur": f"{valeur_point} €"},
                {
                    "Indicateur": "Fenêtre temporelle",
                    "Valeur": f"{year_range[0]}–{year_range[1]}",
                },
                {
                    "Indicateur": "Règle appliquée",
                    "Valeur": "Meilleure position par article (SIGAPS service)",
                },
            ]

            buffer_team = io.BytesIO()
            with pd.ExcelWriter(buffer_team, engine="openpyxl") as writer:
                pd.DataFrame(export_team_rows).to_excel(
                    writer, sheet_name="Articles Equipe", index=False
                )
                pd.DataFrame(indiv_data).to_excel(
                    writer, sheet_name="Contributions Indiv.", index=False
                )
                pd.DataFrame(team_summary).to_excel(
                    writer, sheet_name="Synthèse Equipe", index=False
                )

            st.download_button(
                "⬇️ Télécharger le rapport équipe Excel",
                data=buffer_team.getvalue(),
                file_name=f"SIGAPS_Equipe_{datetime.now().strftime('%Y%m')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ═══════════════════════════════════════════════════════════════
# ONGLET 3 — MÉTHODOLOGIE
# ═══════════════════════════════════════════════════════════════

with tab_methodo:
    st.title("📖 Méthodologie SIGAPS — Algorithme Officiel MERRI 2022")

    st.markdown(
        "<div class='warn-banner'>"
        "⚠️ <b>Avertissement</b> : OpenSIGAPS est un outil d'estimation. La valorisation officielle "
        "est effectuée par la plateforme nationale SIGAPS (<a href='https://www.sigaps.fr'>sigaps.fr</a>). "
        "Les rangs affichés sont des approximations heuristiques à valider."
        "</div>",
        unsafe_allow_html=True,
    )

    st.header("1. Contexte réglementaire")
    st.markdown("""
Le système **SIGAPS** détermine la répartition d'une fraction des **missions MERRI**
(Missions d'Enseignement, de Recherche, de Référence et d'Innovation), soit ~61% de
la dotation socle (circulaire DGOS/R1/2022/110 du 15 avril 2022).

La **réforme 2021** (applicable à partir des MERRI 2022) a introduit quatre changements majeurs :
rang A+, position 3ème auteur, position ADA, et score fractionnaire.

- 📄 [Site SIGAPS officiel](https://www.sigaps.fr)
- 📄 [Document DRCI AP-HP — MERRI 2022](https://recherche-innovation.aphp.fr/wp-content/blogs.dir/77/files/2022/04/SIGAPS-Nouveautes-diffusion-V2.pdf)
- 📄 [DGOS — Financement MERRI](https://solidarites-sante.gouv.fr/IMG/pdf/ds_evolution-modele_v4-2_20210510_pfe.pdf)
    """)

    st.header("2. Les trois modes de calcul")

    st.markdown(
        "<div class='rule-box'>"
        "<b>① Score Présence individuel</b> = C1 × C2<br>"
        "Usage : évaluation carrière hospitalo-universitaire (HU), négociation de poste.<br>"
        "<b>Aucun malus lié au nombre de co-auteurs.</b> C'est le score à mettre en avant "
        "lors d'un entretien de recrutement statutaire."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='rule-box'>"
        "<b>② Score Présence équipe/service</b> = max(C1 membres) × C2 par article unique<br>"
        "Règle officielle : chaque article n'est comptabilisé qu'<b>une seule fois</b> par "
        "structure. Seule la <b>meilleure position</b> parmi les co-auteurs du service est "
        "retenue (Guide pratique SIGAPS, AP-HP).<br>"
        "La somme des scores individuels peut donc dépasser le score service (une co-signature "
        "bénéficie à l'individu mais ne double pas le score du service)."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='rule-box'>"
        "<b>③ Score Fractionnaire MERRI</b> = (C1_auteur / Σ C1_tous_auteurs_article) × C2<br>"
        "Usage : dotation MERRI reversée à l'établissement (depuis 2021).<br>"
        "Le dénominateur Σ C1 porte sur <b>l'ensemble des auteurs de l'article</b> (pas "
        "seulement les membres de l'équipe). Contribution équipe = somme additive des "
        "quotes-parts individuelles."
        "</div>",
        unsafe_allow_html=True,
    )

    st.header("3. Coefficients officiels MERRI 2022")

    # ── Tableaux C1 / C2 en HTML natif (pas de canvas GlideDataGrid) ──
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown(
            """
<div style="margin-bottom:0.4rem;font-family:var(--font-body,sans-serif);font-size:0.72rem;
font-weight:600;text-transform:uppercase;letter-spacing:0.09em;color:var(--gold,#c9a84c);">
Coefficient C1 — Position d'auteur</div>
<table style="width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.85rem;
background:#f8fafc;border:1px solid #dde6f5;border-radius:10px;overflow:hidden;">
<thead>
  <tr style="background:#eef3fb;border-bottom:2px solid rgba(200,146,26,0.35);">
    <th style="padding:10px 14px;text-align:left;color:#c8921a;font-size:0.71rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;">Position</th>
    <th style="padding:10px 14px;text-align:center;color:#c8921a;font-size:0.71rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;">C1</th>
    <th style="padding:10px 14px;text-align:left;color:#c8921a;font-size:0.71rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;">Note</th>
  </tr>
</thead>
<tbody>
  <tr style="border-bottom:1px solid #dde6f5;">
    <td style="padding:9px 14px;color:#1e3a5f;font-weight:500;">1er auteur</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#c8921a;font-size:1rem;">4</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">—</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;background:rgba(37,99,235,0.03);">
    <td style="padding:9px 14px;color:#1e3a5f;font-weight:500;">2ème auteur</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#c8921a;font-size:1rem;">3</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">—</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;">
    <td style="padding:9px 14px;color:#1e3a5f;font-weight:500;">3ème auteur</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#c8921a;font-size:1rem;">2</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">Ajout réforme 2021</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;background:rgba(37,99,235,0.03);">
    <td style="padding:9px 14px;color:#1e3a5f;font-weight:500;">ADA (avant-dernier)</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#c8921a;font-size:1rem;">3</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">Seulement si ≥ 6 auteurs</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;">
    <td style="padding:9px 14px;color:#1e3a5f;font-weight:500;">Dernier auteur</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#c8921a;font-size:1rem;">4</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">—</td>
  </tr>
  <tr>
    <td style="padding:9px 14px;color:#1e3a5f;font-weight:500;background:rgba(37,99,235,0.03);">Autre (intermédiaire)</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#6b7fa3;font-size:1rem;background:rgba(37,99,235,0.03);">1</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;background:rgba(37,99,235,0.03);">—</td>
  </tr>
</tbody>
</table>""",
            unsafe_allow_html=True,
        )

    with col_c2:
        st.markdown(
            """
<div style="margin-bottom:0.4rem;font-family:var(--font-body,sans-serif);font-size:0.72rem;
font-weight:600;text-transform:uppercase;letter-spacing:0.09em;color:var(--gold,#c9a84c);">
Coefficient C2 — Catégorie de la revue</div>
<table style="width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.85rem;
background:#f8fafc;border:1px solid #dde6f5;border-radius:10px;overflow:hidden;">
<thead>
  <tr style="background:#eef3fb;border-bottom:2px solid rgba(200,146,26,0.35);">
    <th style="padding:10px 14px;text-align:center;color:#c8921a;font-size:0.71rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;">Rang</th>
    <th style="padding:10px 14px;text-align:center;color:#c8921a;font-size:0.71rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;">C2</th>
    <th style="padding:10px 14px;text-align:left;color:#c8921a;font-size:0.71rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;">Définition</th>
  </tr>
</thead>
<tbody>
  <tr style="border-bottom:1px solid #dde6f5;background:rgba(200,146,26,0.08);">
    <td style="padding:9px 14px;text-align:center;font-weight:700;color:#c8921a;font-size:0.95rem;
    letter-spacing:0.04em;">A+</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:700;color:#c8921a;font-size:1.1rem;">14</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">NEJM · JAMA · BMJ · Lancet · Nature · Science</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;">
    <td style="padding:9px 14px;text-align:center;font-weight:700;color:#1e3a5f;font-size:0.95rem;">A</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#c8921a;font-size:1.05rem;">8</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">Q1 JIF — top 25% dans la discipline WoS</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;background:rgba(37,99,235,0.03);">
    <td style="padding:9px 14px;text-align:center;font-weight:700;color:#1e3a5f;font-size:0.95rem;">B</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#c8921a;font-size:1.05rem;">6</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">Q2 — 25–50%</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;">
    <td style="padding:9px 14px;text-align:center;font-weight:700;color:#1e3a5f;font-size:0.95rem;">C</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#6b7fa3;font-size:1.05rem;">4</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">Q3 — 50–75%</td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;background:rgba(37,99,235,0.03);">
    <td style="padding:9px 14px;text-align:center;font-weight:700;color:#1e3a5f;font-size:0.95rem;">D</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#6b7fa3;font-size:1.05rem;">3</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">Q4 — 75–100% <span style="color:#48587a">(était 2 avant 2021)</span></td>
  </tr>
  <tr style="border-bottom:1px solid #dde6f5;">
    <td style="padding:9px 14px;text-align:center;font-weight:700;color:#1e3a5f;font-size:0.95rem;">E</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#6b7fa3;font-size:1.05rem;">2</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;">Bas de classement <span style="color:#48587a">(ajout 2021)</span></td>
  </tr>
  <tr>
    <td style="padding:9px 14px;text-align:center;font-weight:700;color:#6b7fa3;font-size:0.95rem;
    background:rgba(37,99,235,0.03);">NC</td>
    <td style="padding:9px 14px;text-align:center;font-family:'JetBrains Mono',monospace;
    font-weight:600;color:#6b7fa3;font-size:1.05rem;background:rgba(37,99,235,0.03);">1</td>
    <td style="padding:9px 14px;color:#6b7fa3;font-size:0.78rem;background:rgba(37,99,235,0.03);">Non classée WoS, indexée PubMed</td>
  </tr>
</tbody>
</table>""",
            unsafe_allow_html=True,
        )

    st.header("4. Matrice des scores présence (C1 × C2)")
    st.caption(
        "Score max : **56 pts** (1er ou Dernier auteur × rang A+). Avant 2021, le maximum était 32 pts."
    )

    # ── Matrice HTML avec dégradé or (pas de canvas Streamlit) ──
    positions = ["1er", "2ème", "3ème", "ADA", "Dernier", "Autre"]
    ranks = ["A+", "A", "B", "C", "D", "E", "NC"]
    max_val = 56.0

    def _cell_style(val: float) -> str:
        ratio = min(val / max_val, 1.0)
        # Dégradé : blanc → bleu navy (thème clair)
        if ratio == 0:
            return "background:#f8fafc;color:#a8b8d8;"
        # Interpolation blanc → #1e3a5f
        r = int(248 - ratio * (248 - 30))
        g = int(250 - ratio * (250 - 58))
        b = int(252 - ratio * (252 - 95))
        lum = 0.299 * r / 255 + 0.587 * g / 255 + 0.114 * b / 255
        text = "#0f1e35" if lum > 0.45 else "#ffffff"
        return f"background:rgb({r},{g},{b});color:{text};"

    header_cells = "".join(
        f'<th style="padding:10px 14px;text-align:center;color:#c8921a;font-size:0.71rem;'
        f'font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">{r}</th>'
        for r in ranks
    )

    rows_html = ""
    for i, pos in enumerate(positions):
        bg_row = "background:rgba(37,99,235,0.035);" if i % 2 else "background:#f8fafc;"
        cells = ""
        for rank in ranks:
            val = SIGAPS_MATRIX.get(pos, {}).get(rank, 0.0)
            cstyle = _cell_style(val)
            mono = "JetBrains Mono, monospace"
            cells += (
                f'<td style="padding:10px 14px;text-align:center;'
                f"font-family:{mono};font-weight:600;font-size:0.95rem;"
                f'{cstyle}">{int(val) if val == int(val) else val}</td>'
            )
        sans = "DM Sans, sans-serif"
        rows_html += (
            f'<tr style="border-bottom:1px solid #dde6f5;{bg_row}">'
            f'<td style="padding:10px 16px;font-weight:600;color:#1e3a5f;'
            f'font-family:{sans};font-size:0.86rem;white-space:nowrap;">{pos}</td>'
            f"{cells}</tr>"
        )

    dm_sans = "DM Sans, sans-serif"
    matrix_html = f"""
<div style="overflow-x:auto;border-radius:10px;border:1px solid #dde6f5;">
<table style="width:100%;border-collapse:collapse;font-family:{dm_sans};background:#f8fafc;">
<thead>
  <tr style="background:#eef3fb;border-bottom:2px solid rgba(200,146,26,0.3);">
    <th style="padding:10px 16px;text-align:left;color:#c8921a;font-size:0.71rem;
    font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">Position / Rang</th>
    {header_cells}
  </tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>
"""
    st.markdown(matrix_html, unsafe_allow_html=True)

    st.header("5. Calcul du dénominateur Σ C1 (score fractionnaire)")
    st.markdown("""
Le dénominateur porte sur **tous** les auteurs de l'article. Exemples officiels (doc. AP-HP) :

| Nb auteurs | Σ C1 | Détail |
|-----------|------|--------|
| 5 | **14** | 4+3+2+1+4 (avant-dernier = Autre car n<6) |
| 6 | **17** | 4+3+2+1+3+4 (ADA activé) |
| 14 | **25** | 4+3+2+9×1+3+4 |
| n ≥ 6 | **n + 11** | Formule générale |

> 💡 Pour un article à 14 auteurs en revue A (C2=8), un 1er auteur contribue :
> (4/25) × 8 = **1,28 pts** au lieu de 4 × 8 = 32 pts en mode présence.
    """)

    st.header("6. Exemple numérique complet")
    st.markdown("""
Un chercheur avec les publications suivantes (fenêtre 4 ans) :

| Position | Rang | Score Présence | Score Frac (n=8 auteurs) |
|----------|------|---------------|--------------------------|
| 1er      | A+   | 4 × 14 = **56 pts** | (4/19) × 14 = **2,95 pts** |
| Dernier  | A    | 4 × 8 = **32 pts**  | (4/19) × 8 = **1,68 pts**  |
| 2ème     | B    | 3 × 6 = **18 pts**  | (3/19) × 6 = **0,95 pts**  |
| ADA      | C    | 3 × 4 = **12 pts**  | (3/19) × 4 = **0,63 pts**  |
| Autre    | C    | 1 × 4 = **4 pts**   | (1/19) × 4 = **0,21 pts**  |

Σ C1 pour n=8 auteurs : 4+3+2+(3×1)+3+4 = 19.

**Total présence = 122 pts** · Valeur annuelle = 122 × 650 = **79 300 €** · Sur 4 ans = **317 200 €**
    """)

    st.header("7. Algorithme d'aide au choix du journal")
    st.markdown("""
L'onglet **🔎 Quel journal choisir ?** intègre un moteur de suggestion sémantique permettant,
à partir du titre d'un article en préparation, d'identifier les journaux SIGAPS les plus pertinents
pour sa soumission.

#### Architecture du pipeline

**Étape 1 — Recherche PubMed (cascade de relaxation)**

Le titre est interrogé sur PubMed via l'API Entrez (esearch), selon une cascade de 5 niveaux progressivement
moins restrictifs :

| Niveau | Requête | Champ | Seuil de déclenchement |
|--------|---------|-------|------------------------|
| 1 | Titre complet | `[Title/Abstract]` | < 15 résultats → niveau suivant |
| 2 | Termes médicaux clés (AND) | `[Title/Abstract]` | < 15 résultats → niveau suivant |
| 3 | 2 termes spécifiques (AND) | Tous champs | < 15 résultats → niveau suivant |
| 4 | Termes clés (OR) | Tous champs | < 15 résultats → niveau suivant |
| 5 | Terme le plus spécifique | Tous champs | Toujours utilisé |

Les stopwords médicaux génériques (*study*, *analysis*, *patient*…) sont exclus de l'extraction.
Jusqu'à **100 articles similaires** sont récupérés. La meilleure récolte est conservée entre les niveaux.

**Étape 2 — Encodage sémantique**

Chaque titre (requête + articles PubMed) est encodé en vecteur dense de 384 dimensions via
**multilingual-e5-small** (Microsoft, 2023), un modèle transformer multilingue compact optimisé
pour la similarité de phrases. En l'absence du modèle, un fallback **Jaccard trigrammes** est utilisé.
Le backend ONNX Runtime (quantification int8) réduit le temps d'inférence de ~30 ms à ~8 ms par phrase.

**Étape 3 — Calcul du score de pertinence**

La similarité cosinus est calculée entre le vecteur du titre soumis et chacun des vecteurs d'articles PubMed :

> **sim(article_i)** = cos(v_requête, v_article_i) = v_requête · v_article_i (vecteurs L2-normalisés)

Les similarités sont agrégées par journal :

> **score_brut(journal_j)** = Σ sim(article_i) pour tous les articles i publiés dans le journal j

Le **score de pertinence affiché (%)** est la normalisation de ce score brut :

> **score_pertinence(j)** = score_brut(j) / max(scores_bruts) × 100

Ce score reflète simultanément la **fréquence** (combien d'articles similaires dans ce journal)
et la **similarité sémantique** (à quel point ces articles ressemblent au titre soumis).

**Étape 4 — Classement final**

Les journaux sont classés par défaut par **rang SIGAPS** (A+ → NC) puis **IF décroissant**, ce
qui permet de cibler en priorité les revues les mieux valorisées. Le tri est modifiable interactivement
(Score de pertinence, IF, Rang seul — ascendant ou descendant).
    """)

    st.header("8. Limites de cet outil")
    st.markdown("""
| Limitation | Impact | Recommandation |
|-----------|--------|---------------|
| **Matching auteur** heuristique | Homonymes possibles | Filtre institution + validation manuelle |
| **Score équipe** sans affiliation officielle | Co-signatures inter-structures non filtrées | Croiser avec les affichages SIGAPS de la DRCI |
| **Nb auteurs** à renseigner manuellement | Score fractionnaire approximatif | Vérifier dans PubMed/DOI |
| **Périmètre** : articles uniquement | Revues, communications exclues | Ajouter manuellement si pertinent |

    """)

    st.header("9. Bibliographie")
    st.markdown("""
- DRCI AP-HP. *Modifications SIGAPS MERRI 2022*. [PDF officiel](https://recherche-innovation.aphp.fr/wp-content/blogs.dir/77/files/2022/04/SIGAPS-Nouveautes-diffusion-V2.pdf)
- Lepage E. *et al.* (2015). SIGAPS score and medical publication evaluation system. *IRBM*, 36(2). [DOI](https://doi.org/10.1016/j.irbm.2015.01.006)
- Priem J. *et al.* (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. *arXiv*. [arXiv:2205.01833](https://arxiv.org/abs/2205.01833)
- Wang L. *et al.* (2024). multilingual-e5-text-embeddings: Multilingual E5 Text Embeddings. *Microsoft Research*. [arXiv:2402.05672](https://arxiv.org/abs/2402.05672)
    """)


# ═══════════════════════════════════════════════════════════════
# ONGLET 4 — QUEL JOURNAL CHOISIR ?
# ═══════════════════════════════════════════════════════════════

with tab_journal:
    st.title("🔎 Quel journal choisir ?")

    _csv_ok_j, _csv_n_j, _ = _CSV_STATUS
    if not _csv_ok_j:
        st.warning(
            "⚠️ **sigaps_ref.csv non chargé** — les rangs affichés seront estimés par heuristique."
        )

    # ── Helpers visuels ────────────────────────────────────────────────────────
    RANK_COLORS = {
        "A+": ("#ffffff", "#c8921a", "rgba(200,146,26,0.13)"),
        "A": ("#0f1e35", "#2563eb", "rgba(37,99,235,0.10)"),
        "B": ("#0f1e35", "#0f766e", "rgba(15,118,110,0.10)"),
        "C": ("#2d4163", "#6b7fa3", "rgba(107,127,163,0.10)"),
        "D": ("#0f1e35", "#b45309", "rgba(180,83,9,0.10)"),
        "E": ("#0f1e35", "#be123c", "rgba(190,18,60,0.10)"),
        "NC": ("#6b7fa3", "#a8b8d8", "rgba(168,184,216,0.12)"),
    }

    def _rank_badge(rank: str, source: str, impact_factor: str = "") -> str:
        txt, border, bg = RANK_COLORS.get(
            rank, ("#dde4f0", "#8a9bbf", "rgba(138,155,191,0.12)")
        )
        if impact_factor:
            src_label = f"IF {impact_factor}"
            src_color = "#b8860b"
        elif source == "csv":
            src_label = "Valeur réelle"
            src_color = "#059669"
        else:
            src_label = "〜 estimé"
            src_color = "#d97706"
        return (
            f'<span style="display:inline-block;padding:4px 12px;border-radius:20px;'
            f"background:{bg};border:1px solid {border};color:{border};"
            f"font-family:JetBrains Mono,monospace;font-weight:700;font-size:1.1rem;"
            f'letter-spacing:0.04em;">{rank}</span>'
            f'&nbsp;<span style="font-size:0.72rem;color:{src_color};'
            f'font-family:DM Sans,sans-serif;font-weight:600;">{src_label}</span>'
        )

    def _score_card(rank: str, position: str, nb_authors: int, valeur: float) -> str:
        p_pts = calculate_presence_score(position, rank)
        f_pts = calculate_fractional_score(position, rank, nb_authors)
        annual = p_pts * valeur
        c1 = C1_COEFFICIENTS.get(position, 1)
        c2 = C2_COEFFICIENTS.get(rank, 1)
        return (
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px;">'
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-top:2px solid #b8860b;border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.12em;color:#6b7fa3;margin-bottom:4px;">C1 × C2</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:1.15rem;'
            f'color:#1e3a5f;">{c1} × {c2}</div></div>'
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-top:2px solid #b8860b;border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.1em;color:#48587a;margin-bottom:4px;">Pts Présence</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:1.6rem;'
            f'font-weight:600;color:#b8860b;">{p_pts:.0f}</div></div>'
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-top:2px solid #2563eb;border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.1em;color:#48587a;margin-bottom:4px;">Pts Fractionnaire</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:1.6rem;'
            f'font-weight:600;color:#2563eb;">{f_pts:.2f}</div></div>'
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-top:2px solid #059669;border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.1em;color:#48587a;margin-bottom:4px;">Valeur annuelle</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:1.6rem;'
            f'font-weight:600;color:#059669;">{annual:,.0f} €</div></div>'
            f"</div>"
        )

    def _journal_card(
        jr,
        position: str,
        nb_authors: int,
        show_similarity: bool = False,
        expanded_calc: bool = False,
        separator: bool = False,
    ) -> None:
        nlm_display = jr.nlm_id or "—"
        issn_display = getattr(jr, "issn", "") or "—"
        country = getattr(jr, "country", "")
        medline_tag = getattr(jr, "medline_indexed", "")

        _left_border = (
            "#b8860b"
            if jr.rank in ("A+", "A")
            else "#2563eb"
            if jr.rank == "B"
            else "#94a3b8"
        )
        _country_span = (
            '<span style="font-size:0.72rem;color:#48587a;">' + country + "</span>"
            if country
            else ""
        )
        _medline_span = ""
        if medline_tag:
            m_color = "#2ec98a" if medline_tag.lower().startswith("o") else "#e05c6e"
            _medline_span = (
                f'<span style="font-size:0.7rem;font-weight:600;color:{m_color};">'
                f"Medline&nbsp;{medline_tag}</span>"
            )

        _sim_bar = ""
        if show_similarity:
            pct = int(jr.similarity * 100)
            bar_color = (
                "#e8cc7a" if pct >= 70 else "#4a90e2" if pct >= 40 else "#8a9bbf"
            )
            art_count = getattr(jr, "article_count", 0)
            _sim_bar = (
                f'<div style="margin-top:10px;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.68rem;color:#94a3b8;margin-bottom:3px;">'
                f"<span>Score de pertinence</span>"
                f'<span style="color:{bar_color};font-weight:700;">{pct}%'
                f"{'&nbsp;·&nbsp;' + str(art_count) + ' article(s) similaire(s)' if art_count else ''}"
                f"</span></div>"
                f'<div style="background:#e2e8f0;border-radius:99px;height:5px;overflow:hidden;">'
                f'<div style="width:{pct}%;height:100%;'
                f'background:{bar_color};border-radius:99px;"></div>'
                f"</div></div>"
            )

        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #dde6f5;'
            f"border-left:4px solid {_left_border}"
            f';border-radius:12px;padding:18px 22px;margin-bottom:8px;box-shadow:0 2px 8px rgba(30,58,95,0.07);">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">'
            f'<div style="flex:1;">'
            f'<div style="font-family:Playfair Display,serif;font-size:1.15rem;'
            f'font-weight:600;color:#1e3a5f;margin-bottom:6px;">{jr.journal_name}</div>'
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;">'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'color:#94a3b8;">Abrév.&nbsp;<b style="color:#64748b;">{jr.medline_ta}</b></span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'color:#94a3b8;">NLM_ID&nbsp;<b style="color:#64748b;">{nlm_display}</b></span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'color:#94a3b8;">ISSN&nbsp;<b style="color:#64748b;">{issn_display}</b></span>'
            f"{_medline_span}{_country_span}"
            f"</div>"
            f"{_sim_bar}"
            f"</div>"
            f'<div style="text-align:right;white-space:nowrap;">'
            f"{_rank_badge(jr.rank, jr.rank_source, jr.impact_factor)}"
            f"</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander(
            f"Score SIGAPS — position : {position} · {nb_authors} auteurs",
            expanded=expanded_calc,
        ):
            c1 = C1_COEFFICIENTS.get(position, 1)
            c2 = C2_COEFFICIENTS.get(jr.rank, 1)
            st.markdown(
                _score_card(jr.rank, position, nb_authors, valeur_point),
                unsafe_allow_html=True,
            )
            st.write("")
            st.caption(
                f"Position **{position}** · Rang **{jr.rank}** · "
                f"C1={c1} · C2={c2} · {nb_authors} auteurs · Valeur point : {valeur_point} €"
            )

        # ── Graphique historique rang + IF (si données disponibles) ──
        _hist = _ref_db.get_history(jr.nlm_id) if _ref_db and jr.nlm_id else {}
        if _hist:
            with st.expander(
                f"Évolution rang & IF — {jr.journal_name}", expanded=False
            ):
                _years = sorted(_hist.keys())
                _ranks = [_hist[y].get("rank", None) for y in _years]
                _ifs = []
                for y in _years:
                    try:
                        _ifs.append(float(_hist[y]["if"].replace(",", ".")))
                    except:
                        _ifs.append(None)
                _RANK_NUM = {"A+": 7, "A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "NC": 1}
                _rank_nums = [_RANK_NUM.get(r, None) for r in _ranks]
                _has_rank = any(v is not None for v in _rank_nums)
                _has_if = any(v is not None for v in _ifs)
                if _has_rank or _has_if:
                    _nrows = 2 if (_has_rank and _has_if) else 1
                    _specs = [[{"secondary_y": False}]] * _nrows
                    _titles = []
                    if _has_rank:
                        _titles.append("Rang SIGAPS")
                    if _has_if:
                        _titles.append("Impact Factor")
                    _fig = make_subplots(
                        rows=_nrows,
                        cols=1,
                        subplot_titles=_titles,
                        vertical_spacing=0.18,
                    )
                    _row = 1
                    _RANK_LABEL = {
                        7: "A+",
                        6: "A",
                        5: "B",
                        4: "C",
                        3: "D",
                        2: "E",
                        1: "NC",
                    }
                    _RANK_COLOR = {
                        "A+": "#c8921a",
                        "A": "#2563eb",
                        "B": "#0f766e",
                        "C": "#6b7fa3",
                        "D": "#b45309",
                        "E": "#be123c",
                        "NC": "#a8b8d8",
                    }
                    if _has_rank:
                        _colors = [_RANK_COLOR.get(r, "#6b7fa3") for r in _ranks]
                        _fig.add_trace(
                            go.Scatter(
                                x=_years,
                                y=_rank_nums,
                                mode="lines+markers+text",
                                text=_ranks,
                                textposition="top center",
                                textfont=dict(
                                    size=11, color="#1e3a5f", family="JetBrains Mono"
                                ),
                                line=dict(color="#2563eb", width=2.5),
                                marker=dict(
                                    size=10,
                                    color=_colors,
                                    line=dict(color="#1e3a5f", width=1.5),
                                ),
                                name="Rang",
                                hovertemplate="%{text}<extra></extra>",
                            ),
                            row=_row,
                            col=1,
                        )
                        _fig.update_yaxes(
                            tickvals=list(_RANK_NUM.values()),
                            ticktext=list(_RANK_NUM.keys()),
                            range=[0, 8],
                            row=_row,
                            col=1,
                        )
                        _row += 1
                    if _has_if:
                        _if_vals = [v for v in _ifs if v is not None]
                        _fig.add_trace(
                            go.Scatter(
                                x=_years,
                                y=_ifs,
                                mode="lines+markers",
                                line=dict(color="#c8921a", width=2.5),
                                marker=dict(
                                    size=8,
                                    color="#c8921a",
                                    line=dict(color="#1e3a5f", width=1.5),
                                ),
                                name="IF",
                                fill="tozeroy",
                                fillcolor="rgba(200,146,26,0.08)",
                                hovertemplate="IF %{y:.2f}<extra></extra>",
                            ),
                            row=_row,
                            col=1,
                        )
                    _fig.update_layout(
                        height=260 * _nrows,
                        margin=dict(t=40, b=20, l=30, r=10),
                        paper_bgcolor="#f8fafc",
                        plot_bgcolor="#f8fafc",
                        font=dict(family="Sora, sans-serif", color="#1e3a5f", size=12),
                        showlegend=False,
                        xaxis=dict(dtick=1, gridcolor="#dde6f5"),
                        xaxis2=dict(dtick=1, gridcolor="#dde6f5") if _nrows > 1 else {},
                    )
                    _fig.update_xaxes(gridcolor="#dde6f5")
                    _fig.update_yaxes(gridcolor="#dde6f5")
                    st.plotly_chart(
                        _fig, use_container_width=True, config={"displayModeBar": False}
                    )
                else:
                    st.caption(
                        "Données historiques insuffisantes dans le référentiel CSV."
                    )

        if separator:
            st.markdown(
                '<div style="height:1px;background:#dde6f5;margin:6px 0 12px;"></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # PARAMÈTRES COMMUNS — saisis une seule fois, partagés par les 2 sous-onglets
    # ══════════════════════════════════════════════════════════════════════════

    _pc1, _pc2, _pc3 = st.columns([2, 2, 4])
    with _pc1:
        shared_position = st.selectbox(
            "Votre position",
            ALL_POSITIONS,
            index=0,
            key="shared_pos",
        )
    with _pc2:
        shared_nb_authors = st.number_input(
            "Nombre d'auteurs",
            min_value=1,
            max_value=200,
            value=5,
            step=1,
            key="shared_nb",
            help="Nécessaire pour le score fractionnaire MERRI.",
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SOUS-ONGLETS
    # ══════════════════════════════════════════════════════════════════════════

    sub_tab_know, sub_tab_suggest = st.tabs(
        [
            "🔤 J'ai une idée de journal",
            "🧠 Aide au choix d'un journal",
        ]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SOUS-ONGLET A — J'AI UNE IDÉE DE JOURNAL
    # Recherche par titre de journal ou PMID
    # ──────────────────────────────────────────────────────────────────────────

    with sub_tab_know:
        search_mode = st.radio(
            "Mode",
            ["Par titre de journal", "Par DOI d'article"],
            horizontal=True,
            label_visibility="collapsed",
            key="sub_a_mode",
        )

        if search_mode == "Par titre de journal":
            journal_query = st.text_input(
                "Titre ou mot-clé du journal",
                placeholder="Ex: Annals of Oncology · Blood · Frontiers in Medicine",
                key="jq_title",
            )
            doi_query = ""
        else:
            doi_query = st.text_input(
                "DOI de l'article",
                placeholder="Ex: 10.1182/blood.2023021234",
                key="jq_doi",
            )
            journal_query = ""

        search_journal_btn = st.button(
            "🔍 Rechercher",
            key="btn_jrnl_search",
            disabled=(not journal_query.strip() and not doi_query.strip()),
            help=(
                "Recherche dans le NLM Catalog (API Entrez) et votre référentiel SIGAPS. "
                "Retourne le rang SIGAPS, l'IF et les métadonnées du journal. "
                "Fonctionne par titre de journal (ex: Blood, Annals of Oncology) ou par PMID d'article existant."
            ),
        )

        if search_journal_btn:
            st.divider()

            # ── MODE PMID ─────────────────────────────────────────────────────
            if doi_query.strip():
                with st.spinner("Résolution DOI via PubMed…"):
                    art_data = fetch_article_by_doi(doi_query.strip(), ref_db=_ref_db)

                if art_data is None:
                    st.error(
                        f"DOI `{doi_query}` introuvable sur PubMed. Vérifiez le format (ex: 10.1182/blood.2023021234)."
                    )
                else:
                    st.subheader("Article trouvé")
                    nlm_display = art_data["nlm_id"] or "Non disponible"
                    _doi_part = (
                        "&nbsp;·&nbsp;" + art_data["doi"] if art_data.get("doi") else ""
                    )
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #dde6f5;'
                        f'border-radius:12px;padding:20px 24px;margin-bottom:8px;box-shadow:0 2px 10px rgba(30,58,95,0.08);">'
                        f'<div style="font-family:Playfair Display,serif;font-size:1.25rem;'
                        f'color:#1e3a5f;font-weight:600;line-height:1.4;margin-bottom:10px;">'
                        f"{art_data['title']}</div>"
                        f'<div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;">'
                        f'<span style="font-size:0.82rem;color:#6b7fa3;">'
                        f'📰 <b style="color:#dde4f0;">{art_data["journal"]}</b></span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.78rem;'
                        f'color:#6b7fa3;">NLM_ID&nbsp;{nlm_display}</span>'
                        f'<span style="font-size:0.78rem;color:#48587a;">'
                        f"{art_data.get('year', '—')}{_doi_part}</span>"
                        f"</div>"
                        f"<div>Rang SIGAPS : {_rank_badge(art_data['rank'], art_data['rank_source'], art_data.get('impact_factor', ''))}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Score pour votre position**")
                    st.markdown(
                        _score_card(
                            art_data["rank"],
                            shared_position,
                            shared_nb_authors,
                            valeur_point,
                        ),
                        unsafe_allow_html=True,
                    )
                    if art_data.get("authors"):
                        with st.expander("Auteurs"):
                            st.markdown(
                                f'<p style="font-family:DM Sans,sans-serif;font-size:0.82rem;'
                                f'color:#8a9bbf;line-height:1.8;">'
                                f"{' · '.join(art_data['authors'][:20])}</p>",
                                unsafe_allow_html=True,
                            )

            # ── MODE TITRE JOURNAL ────────────────────────────────────────────
            elif journal_query.strip():
                with st.spinner("Recherche NLM Catalog + référentiel CSV…"):
                    journal_results = search_journal_by_name(
                        journal_query.strip(), ref_db=_ref_db, max_results=8
                    )

                if not journal_results:
                    st.error(
                        f"Aucun journal trouvé pour « {journal_query} ». "
                        "Essayez un terme plus court ou en anglais."
                    )
                else:
                    nb_j = len(journal_results)
                    st.subheader(
                        f"{nb_j} journal{'x' if False else 'aux' if nb_j > 1 else ''} "
                        f"trouvé{'s' if nb_j > 1 else ''}"
                    )
                    for i, jr in enumerate(journal_results):
                        _journal_card(
                            jr,
                            shared_position,
                            shared_nb_authors,
                            show_similarity=False,
                            expanded_calc=(i == 0),
                            separator=(i < nb_j - 1),
                        )

    # ──────────────────────────────────────────────────────────────────────────
    # SOUS-ONGLET B — AIDE AU CHOIX D'UN JOURNAL (NLP)
    # Suggestion par titre d'article avec multilingual-e5-small
    # ──────────────────────────────────────────────────────────────────────────

    with sub_tab_suggest:
        article_title_input = st.text_area(
            "Titre de votre article",
            placeholder=(
                "Ex: Effect of Deferasirox Six Months after allo-HSCT on AML/MDS Outcomes: "
                "a Propensity-Score Matched Study"
            ),
            height=90,
            key="nlp_title",
        )

        # ── Bouton de recherche ───────────────────────────────────────────────
        suggest_btn = st.button(
            "🧠 Trouver des journaux cibles",
            key="btn_suggest",
            disabled=not article_title_input.strip(),
            help=(
                "Interroge PubMed (jusqu'à 100 articles) + analyse sémantique NLP. "
                f"Moteur : {_embed_backend.upper() if _embed_model else 'Jaccard'}."
            ),
        )

        # ── Lancement de la recherche (stockée en session_state) ──────────────
        if suggest_btn and article_title_input.strip():
            _cache_key = article_title_input.strip()
            if _cache_key != st.session_state.suggest_query_cache:
                with st.spinner(
                    "Interrogation PubMed (jusqu'à 100 articles) + analyse sémantique…"
                ):
                    st.session_state.raw_suggestions = suggest_journals_by_title(
                        _cache_key,
                        ref_db=_ref_db,
                        max_articles=100,
                        max_journals=30,
                    )
                st.session_state.suggest_query_cache = _cache_key

        # ── Résultats (re-évalués à chaque changement de filtre/tri) ──────────
        raw_suggestions = st.session_state.raw_suggestions

        if not raw_suggestions and st.session_state.suggest_query_cache:
            st.error(
                "Aucun article similaire trouvé sur PubMed pour ce titre. "
                "Essayez avec quelques mots-clés principaux uniquement."
            )

        if raw_suggestions:
            # ── Bannière niveau de relaxation ─────────────────────────────────
            ql = raw_suggestions[0].query_level
            qu = raw_suggestions[0].query_used
            _ql_icon, _ql_color, _ql_msg = CASCADE_LEVEL_LABELS.get(
                ql, ("⚠️", "#e05c6e", f"Niveau de relaxation {ql}")
            )
            st.markdown(
                f'<div style="background:#f4f7fb;border:1px solid #dde6f5;'
                f"border-left:3px solid {_ql_color};border-radius:6px;"
                f'padding:7px 14px;margin-bottom:10px;font-size:0.79rem;display:flex;align-items:center;gap:8px;">'
                f"<span>{_ql_icon}</span>"
                f'<b style="color:{_ql_color};">{_ql_msg}</b>'
                f'<span style="color:#a8b8d8;font-family:JetBrains Mono,monospace;'
                f'font-size:0.70rem;margin-left:4px;">{qu}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── BARRE FILTRES/TRI COMPACTE (inline, réactive) ─────────────────
            # Les widgets Streamlit mis en columns ultra-compacts.
            # Pas besoin de relancer : session_state.raw_suggestions persiste.
            st.markdown(
                '<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">'
                '<span style="font-size:0.65rem;font-weight:700;color:#6b7fa3;'
                'text-transform:uppercase;letter-spacing:0.09em;white-space:nowrap;">Filtres</span>'
                '<div style="flex:1;height:1px;background:#dde6f5;"></div></div>',
                unsafe_allow_html=True,
            )
            _f1, _f2, _f3, _fsep, _f4, _f5 = st.columns([2.8, 1.1, 1.0, 0.1, 2.0, 0.7])
            with _f1:
                filter_ranks = st.multiselect(
                    "Rangs SIGAPS",
                    ALL_RANKS,
                    default=ALL_RANKS,
                    key="nlp_filter_ranks",
                    placeholder="Tous les rangs…",
                )
            with _f2:
                filter_if_min = st.number_input(
                    "IF min",
                    min_value=0.0,
                    max_value=200.0,
                    value=0.0,
                    step=0.5,
                    key="nlp_filter_if",
                    help="IF minimum (0 = sans filtre)",
                )
            with _f3:
                filter_medline_only = st.checkbox(
                    "Medline only",
                    value=True,
                    key="nlp_filter_medline",
                )
            with _fsep:
                st.markdown(
                    '<div style="height:38px;width:1px;background:#dde6f5;margin:auto;"></div>',
                    unsafe_allow_html=True,
                )
            with _f4:
                _sort_by = st.selectbox(
                    "Trier par",
                    ["Rang SIGAPS puis IF", "Score de pertinence", "IF", "Rang seul"],
                    index=0,
                    key="suggest_sort_by",
                )
            with _f5:
                _sort_asc = st.checkbox("↑ Asc", value=False, key="suggest_sort_asc")

            # ── Application des filtres ───────────────────────────────────────
            def _if_float(s) -> float:
                try:
                    return float(s.impact_factor.replace(",", "."))
                except (ValueError, AttributeError):
                    return 0.0

            filtered = list(raw_suggestions)
            if filter_ranks:
                filtered = [s for s in filtered if s.rank in filter_ranks]
            if filter_if_min > 0:
                filtered = [s for s in filtered if _if_float(s) >= filter_if_min]
            if filter_medline_only:
                filtered = [
                    s for s in filtered if s.medline_indexed.lower().startswith("o")
                ]

            if not filtered:
                st.warning(
                    "Aucun journal ne correspond aux filtres actifs. "
                    "Élargissez les critères (rangs, IF minimum, Medline)."
                )
            else:
                # ── Tri ───────────────────────────────────────────────────────
                _RANK_ORD = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "NC": 6}
                if _sort_by == "Rang SIGAPS puis IF":
                    filtered.sort(
                        key=lambda s: (
                            _RANK_ORD.get(s.rank, 7) * (1 if not _sort_asc else -1),
                            -_if_float(s) * (1 if not _sort_asc else -1),
                        )
                    )
                elif _sort_by == "Score de pertinence":
                    filtered.sort(key=lambda s: s.similarity, reverse=not _sort_asc)
                elif _sort_by == "IF":
                    filtered.sort(key=lambda s: _if_float(s), reverse=not _sort_asc)
                else:  # Rang seul
                    filtered.sort(
                        key=lambda s: _RANK_ORD.get(s.rank, 7), reverse=_sort_asc
                    )

                # ── En-tête résultats + export Excel ─────────────────────────
                total_arts = sum(s.article_count for s in raw_suggestions)
                nb_s = len(filtered)
                _hcol1, _hcol2 = st.columns([4, 1])
                with _hcol1:
                    st.markdown(
                        f'<div style="font-family:var(--font-display);font-size:1.05rem;'
                        f'font-weight:600;color:var(--navy);margin:6px 0 4px;">'
                        f"{nb_s} journal{'aux' if nb_s > 1 else ''} suggéré{'s' if nb_s > 1 else ''}"
                        f"</div>"
                        f'<div style="font-size:0.72rem;color:var(--text-muted);font-family:var(--font-body);margin-bottom:12px;">'
                        f"{total_arts} articles PubMed analysés · "
                        f"Medline : {'actif' if filter_medline_only else 'inactif'} · "
                        f"Tri : {_sort_by} {'↑' if _sort_asc else '↓'}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with _hcol2:
                    _export_rows = [
                        {
                            "Rang": s.rank,
                            "Journal": s.journal_name,
                            "Abréviation": s.medline_ta,
                            "NLM_ID": s.nlm_id,
                            "ISSN": getattr(s, "issn", ""),
                            "IF": getattr(s, "impact_factor", ""),
                            "Medline": getattr(s, "medline_indexed", ""),
                            "Score pertinence (%)": int(s.similarity * 100),
                            "Articles similaires": s.article_count,
                        }
                        for s in filtered[:30]
                    ]
                    _buf_sug = io.BytesIO()
                    with pd.ExcelWriter(_buf_sug, engine="openpyxl") as _xw:
                        pd.DataFrame(_export_rows).to_excel(
                            _xw, sheet_name="Suggestions", index=False
                        )
                    st.download_button(
                        "⬇️ Excel",
                        data=_buf_sug.getvalue(),
                        file_name=f"SIGAPS_suggestions_{datetime.now().strftime('%Y%m')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                # ── Affichage des résultats — chaque journal dans un bloc unifié ──
                for i, sug in enumerate(filtered[:30]):
                    # ══ SÉPARATEUR MARQUÉ entre chaque résultat ══
                    if i > 0:
                        st.markdown(
                            '<div style="margin:28px 0 20px;display:flex;align-items:center;gap:12px;">'
                            '<div style="flex:1;height:2px;'
                            "background:linear-gradient(90deg,#c8d8ee 0%,#dde6f5 100%);"
                            'border-radius:99px;"></div>'
                            f'<span style="font-size:0.62rem;font-weight:700;color:#a8b8d8;'
                            f'text-transform:uppercase;letter-spacing:0.12em;white-space:nowrap;">'
                            f"— Résultat {i + 1} —</span>"
                            '<div style="flex:1;height:2px;'
                            "background:linear-gradient(90deg,#dde6f5 0%,#c8d8ee 100%);"
                            'border-radius:99px;"></div></div>',
                            unsafe_allow_html=True,
                        )

                    # ══ BLOC RÉSULTAT — carte + expanders groupés visuellement ══
                    # Enveloppe avec bordure gauche colorée selon le rang
                    _rank_left = {
                        "A+": "#c8921a",
                        "A": "#c8921a",
                        "B": "#2563eb",
                        "C": "#6b7fa3",
                        "D": "#b45309",
                        "E": "#be123c",
                        "NC": "#a8b8d8",
                    }.get(sug.rank, "#dde6f5")

                    st.markdown(
                        f'<div style="border-left:4px solid {_rank_left};'
                        f"border-radius:0 12px 12px 0;"
                        f'padding-left:12px;margin-bottom:0;">',
                        unsafe_allow_html=True,
                    )

                    _journal_card(
                        sug,
                        shared_position,
                        shared_nb_authors,
                        show_similarity=True,
                        expanded_calc=False,
                        separator=False,
                    )

                    # ── Articles similaires — immédiatement sous la carte ────
                    if sug.example_titles:
                        with st.expander(
                            f"Articles similaires dans ce journal ({sug.article_count})",
                            expanded=False,
                        ):
                            for art in sug.example_titles:
                                if isinstance(art, dict):
                                    _t = art.get("title", "")
                                    _pmid = art.get("pmid", "")
                                else:
                                    _t = str(art)
                                    _pmid = ""
                                _url = (
                                    f"https://pubmed.ncbi.nlm.nih.gov/{_pmid}/"
                                    if _pmid
                                    else ""
                                )
                                if _url:
                                    _link_html = (
                                        f'<a href="{_url}" target="_blank" rel="noopener" '
                                        f'style="flex-shrink:0;display:inline-flex;align-items:center;'
                                        f"gap:3px;font-size:0.68rem;color:#2563eb;"
                                        f"font-family:JetBrains Mono,monospace;font-weight:600;"
                                        f"text-decoration:none;border:1px solid rgba(37,99,235,0.22);"
                                        f"border-radius:4px;padding:2px 7px;"
                                        f'background:rgba(37,99,235,0.06);">'
                                        f"↗ PMID {_pmid}</a>"
                                    )
                                else:
                                    _link_html = ""
                                st.markdown(
                                    f'<div style="display:flex;align-items:flex-start;'
                                    f"justify-content:space-between;gap:14px;"
                                    f'padding:10px 6px;border-bottom:1px solid #eef3fb;">'
                                    f'<span style="font-size:0.83rem;color:#2d4163;'
                                    f'font-family:Sora,sans-serif;line-height:1.55;flex:1;">{_t}</span>'
                                    f"{_link_html}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # Fermeture visuelle du bloc (le div border-left n'est pas fermable
                    # en Streamlit, donc on ajoute juste un espace de respiration)
                    st.markdown(
                        '<div style="margin-bottom:4px;"></div>', unsafe_allow_html=True
                    )
