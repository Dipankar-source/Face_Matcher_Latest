# ============================
# 1. COMPATIBILITY & PATCHES
# ============================
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys, types
from tensorflow.keras.utils import get_source_inputs

keras_engine          = types.ModuleType("keras.engine")
keras_engine_topology = types.ModuleType("keras.engine.topology")
keras_engine_topology.get_source_inputs = get_source_inputs
sys.modules["keras.engine"]          = keras_engine
sys.modules["keras.engine.topology"] = keras_engine_topology

# ============================
# 2. IMPORTS
# ============================
import cv2, pickle, numpy as np, shutil, uuid
import streamlit as st
import pandas as pd
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras_vggface.vggface import VGGFace
from keras_vggface.utils  import preprocess_input

if not hasattr(st, "rerun"):
    st.rerun = st.experimental_rerun

# ============================
# 3. PAGE CONFIG
# ============================
st.set_page_config(layout="wide", page_title="FaceID Pro", page_icon="◈")

# ============================
# 4. CSS
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg:        #08080b;
    --bg1:       #0f0f14;
    --bg2:       #16161d;
    --border:    rgba(255,255,255,0.055);
    --border-hi: rgba(255,255,255,0.11);
    --accent:    #7b6cff;
    --accent2:   #00cfb4;
    --text:      #eceaf8;
    --muted:     #5a5968;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Space Grotesk', sans-serif;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 24px 80px !important;
    min-height: 100vh;
}

/* dashed grid background */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(var(--border) 1px, transparent 1px),
        linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(circle, rgba(123,108,255,0.13) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}
.stApp > * { position: relative; z-index: 1; }

/* masthead */
.masthead {
    display: flex; align-items: center; justify-content: space-between;
    padding: 22px 0 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 34px;
}
.masthead-logo {
    font-family: var(--mono); font-size: 12px; font-weight: 500;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--accent); display: flex; align-items: center; gap: 9px;
}
.masthead-logo span { font-size: 17px; }
.masthead-tag {
    font-family: var(--mono); font-size: 10px; color: var(--muted);
    letter-spacing: 0.1em; border: 1px solid var(--border-hi);
    padding: 3px 9px; border-radius: 3px;
}

/* upload zone */
.upload-eyebrow {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.22em;
    color: var(--muted); text-align: center;
    margin-bottom: 10px; text-transform: uppercase;
    display: flex; align-items: center; justify-content: center; gap: 10px;
}
.upload-eyebrow::before,
.upload-eyebrow::after {
    content: ''; display: inline-block;
    width: 32px; height: 1px; background: var(--border-hi);
}

div[data-testid="stFileUploader"] {
    max-width: 480px !important;
    margin: 0 auto 36px !important;
}
section[data-testid="stFileUploaderDropzone"] {
    background: var(--bg1) !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 160px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: background .25s !important;
    box-shadow:
        inset 0 0 0 1px rgba(123,108,255,0.22),
        0 0 0 1px var(--border) !important;
}
section[data-testid="stFileUploaderDropzone"]::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 22px; height: 22px;
    border-top: 2px solid rgba(123,108,255,0.60);
    border-left: 2px solid rgba(123,108,255,0.60);
    border-radius: 12px 0 0 0;
    transition: width .28s, height .28s, border-color .25s;
    pointer-events: none; z-index: 2;
}
section[data-testid="stFileUploaderDropzone"]::after {
    content: ''; position: absolute; bottom: 0; right: 0;
    width: 22px; height: 22px;
    border-bottom: 2px solid rgba(123,108,255,0.60);
    border-right: 2px solid rgba(123,108,255,0.60);
    border-radius: 0 0 12px 0;
    transition: width .28s, height .28s, border-color .25s;
    pointer-events: none; z-index: 2;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(123,108,255,0.035) !important;
    box-shadow:
        inset 0 0 0 1px rgba(123,108,255,0.45),
        0 0 24px rgba(123,108,255,0.07) !important;
}
section[data-testid="stFileUploaderDropzone"]:hover::before {
    width: 44px; height: 44px; border-color: var(--accent);
}
section[data-testid="stFileUploaderDropzone"]:hover::after {
    width: 44px; height: 44px; border-color: var(--accent);
}
@keyframes scanline {
    0%   { top: -4px; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
section[data-testid="stFileUploaderDropzone"]:hover > div::before {
    content: '';
    position: absolute; left: 0; right: 0; top: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, rgba(123,108,255,0.0) 10%,
        rgba(123,108,255,0.55) 50%,
        rgba(0,207,180,0.35) 80%, transparent 100%);
    animation: scanline 1.6s ease-in-out infinite;
    pointer-events: none; z-index: 3;
}
section[data-testid="stFileUploaderDropzone"] p,
section[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stFileUploaderDropzone"] small {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--bg2) !important;
    border: 1px solid rgba(123,108,255,0.28) !important;
    color: rgba(123,108,255,0.85) !important;
    font-family: var(--mono) !important;
    font-size: 9px !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 5px 12px !important;
    margin-top: 8px !important;
    transition: background .18s, border-color .18s, color .18s !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(123,108,255,0.10) !important;
    border-color: rgba(123,108,255,0.55) !important;
    color: #c0b8ff !important;
}

/* section labels */
.sec-label {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.2em;
    color: var(--muted); text-transform: uppercase;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}
.sec-label::before {
    content: ''; display: inline-block;
    width: 14px; height: 1px; background: var(--accent);
}

/* image frame */
.img-frame {
    border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden; background: var(--bg1);
    transition: border-color .22s, transform .22s;
}
.img-frame:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.img-frame-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 12px;
    border-bottom: 1px dashed var(--border-hi);
    background: rgba(255,255,255,0.018);
}
.img-frame-title {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase;
}
.img-frame-dot { width: 5px; height: 5px; border-radius: 50%; }
.img-frame-body {
    padding: 10px;
    display: flex; align-items: center; justify-content: center;
}

/* images */
div[data-testid="stImage"] {
    display: flex !important; justify-content: center !important;
}
div[data-testid="stImage"] img {
    max-width: 100% !important;
    width: auto !important; height: auto !important;
    border-radius: 5px !important;
    transition: transform .32s cubic-bezier(.22,1,.36,1), filter .28s !important;
    filter: brightness(.94) saturate(.88) !important;
}
div[data-testid="stImage"]:hover img {
    transform: scale(1.032) !important;
    filter: brightness(1) saturate(1.05) !important;
}

/* result card */
.result-card {
    border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden; background: var(--bg1); margin-bottom: 14px;
    transition: border-color .22s, transform .2s;
}
.result-card:hover { border-color: var(--border-hi); transform: translateY(-1px); }
.rc-header {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 14px;
    border-bottom: 1px dashed var(--border-hi);
    background: rgba(255,255,255,0.018);
}
.rc-header-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent2); }
.rc-header-label {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.18em; color: var(--muted); text-transform: uppercase;
}
.rc-body { padding: 18px 16px; }
.id-name {
    font-family: var(--sans); font-size: 30px; font-weight: 700;
    letter-spacing: -0.02em; color: var(--text);
    margin: 0 0 2px; line-height: 1.1;
}
.id-sub { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.1em; }
.score-row {
    display: flex; align-items: baseline; justify-content: space-between;
    margin: 18px 0 7px;
}
.score-lbl {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase;
}
.score-val {
    font-family: var(--mono); font-size: 22px; font-weight: 500;
    color: var(--dyn-color);
}
.score-track {
    height: 2px; background: var(--border);
    border-radius: 2px; overflow: hidden; margin-bottom: 12px;
}
.score-fill {
    height: 100%; border-radius: 2px;
    width: var(--dyn-pct);
    background: var(--dyn-color);
}
.status-chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: var(--mono); font-size: 10px; font-weight: 500;
    letter-spacing: 0.14em; padding: 3px 9px; border-radius: 3px;
    text-transform: uppercase;
    color: var(--dyn-color);
    background: var(--dyn-chip-bg);
    border: 1px solid var(--dyn-chip-border);
}
.status-chip::before {
    content: ''; width: 4px; height: 4px;
    border-radius: 50%; background: currentColor;
}

/* ── EVALUATION PANEL ─────────────────────────────────── */
.eval-section-title {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.25em;
    text-transform: uppercase; color: var(--muted);
    margin: 36px 0 20px;
    display: flex; align-items: center; gap: 12px;
}
.eval-section-title::before { content: ''; flex: none; width: 14px; height: 1px; background: var(--accent); }
.eval-section-title::after  { content: ''; flex: 1;    height: 1px; background: var(--border); }

/* stat tiles */
.eval-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.eval-tile {
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 14px 14px;
    transition: border-color .2s, transform .2s;
}
.eval-tile:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.eval-tile-label {
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}
.eval-tile-val {
    font-family: var(--mono); font-size: 26px; font-weight: 500;
    color: var(--text); line-height: 1;
}
.eval-tile-sub {
    font-family: var(--mono); font-size: 9px;
    color: var(--muted); margin-top: 4px;
}
.eval-tile-bar {
    height: 2px; background: var(--border);
    border-radius: 2px; overflow: hidden; margin-top: 10px;
}
.eval-tile-bar-fill {
    height: 100%; border-radius: 2px;
}

/* classification report table */
.eval-table-wrap {
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 24px;
}
.eval-table-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px;
    border-bottom: 1px dashed var(--border-hi);
    background: rgba(255,255,255,0.018);
}
.eval-table-title {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--muted);
}
.eval-table-badge {
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.1em;
    color: var(--accent2);
    border: 1px solid rgba(0,207,180,0.22); padding: 2px 8px; border-radius: 3px;
}

/* override streamlit dataframe styling */
div[data-testid="stDataFrame"] {
    background: transparent !important;
    border: none !important;
}
div[data-testid="stDataFrame"] table {
    background: transparent !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    width: 100% !important;
}
div[data-testid="stDataFrame"] th {
    background: rgba(123,108,255,0.06) !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 9px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border-hi) !important;
    padding: 8px 12px !important;
}
div[data-testid="stDataFrame"] td {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 7px 12px !important;
}
div[data-testid="stDataFrame"] tr:hover td {
    background: rgba(123,108,255,0.035) !important;
}

/* confusion matrix table */
.cm-wrap {
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: auto;
    margin-bottom: 24px;
}
.cm-inner { padding: 16px; }
.cm-table {
    border-collapse: collapse;
    width: 100%;
    font-family: var(--mono);
    font-size: 11px;
}
.cm-table th {
    padding: 6px 10px;
    background: rgba(123,108,255,0.07);
    color: var(--muted);
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-hi);
    text-align: center;
    white-space: nowrap;
}
.cm-table th.row-header { text-align: left; }
.cm-table td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    text-align: center;
    color: var(--text);
    transition: background .15s;
}
.cm-table td.label-cell {
    color: var(--muted);
    font-size: 9px;
    letter-spacing: 0.1em;
    text-align: left;
    white-space: nowrap;
    background: rgba(255,255,255,0.012);
    border-right: 1px dashed var(--border-hi);
}
.cm-table tr:last-child td { border-bottom: none; }
.cm-cell-high  { color: #00e5a0 !important; font-weight: 500; }
.cm-cell-med   { color: #f0a500 !important; }
.cm-cell-zero  { color: var(--muted) !important; font-size: 9px; }
.cm-diagonal   { background: rgba(0,207,180,0.07) !important; }

/* accuracy banner */
.acc-banner {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg1); border: 1px solid rgba(0,207,180,0.22);
    border-radius: 10px; padding: 16px 20px; margin-bottom: 24px;
}
.acc-banner-label {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--muted);
}
.acc-banner-value {
    font-family: var(--mono); font-size: 38px; font-weight: 500;
    color: #00e5a0; line-height: 1;
}
.acc-banner-sub {
    font-family: var(--mono); font-size: 9px; color: var(--muted);
    letter-spacing: 0.1em; margin-top: 2px;
}
.acc-track {
    flex: 1; height: 2px; background: var(--border);
    border-radius: 2px; overflow: hidden; margin: 0 28px;
}
.acc-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #00cfb4, #00e5a0);
}

/* sample info row */
.eval-meta-row {
    display: flex; gap: 12px; margin-bottom: 20px;
}
.eval-meta-chip {
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.14em;
    color: var(--muted); text-transform: uppercase;
    border: 1px solid var(--border-hi); padding: 4px 10px; border-radius: 3px;
    display: flex; align-items: center; gap: 5px;
}
.eval-meta-chip span { color: var(--text); font-size: 10px; }

/* buttons */
div[data-testid="stButton"] button {
    background: var(--bg2) !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important; font-size: 10px !important;
    letter-spacing: 0.14em !important; border-radius: 5px !important;
    padding: 9px 14px !important; text-transform: uppercase !important;
    transition: all .18s !important;
}
div[data-testid="stButton"] button:hover {
    background: rgba(123,108,255,0.09) !important;
    border-color: rgba(123,108,255,0.42) !important;
    color: #c0b8ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(123,108,255,0.1) !important;
}
div[data-testid="stButton"] button:active { transform: scale(.98) !important; }
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg,#7b6cff,#5a4fdb) !important;
    border: none !important; color: #fff !important;
    font-family: var(--mono) !important; font-size: 10px !important;
    letter-spacing: 0.14em !important; border-radius: 5px !important;
    padding: 9px 14px !important; text-transform: uppercase !important;
    transition: all .18s !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    opacity: .88 !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(123,108,255,0.28) !important;
}

/* text input */
div[data-testid="stTextInput"] input {
    background: var(--bg) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 5px !important; color: var(--text) !important;
    font-family: var(--mono) !important; font-size: 12px !important;
    transition: border-color .18s, box-shadow .18s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(123,108,255,0.52) !important;
    box-shadow: 0 0 0 3px rgba(123,108,255,0.07) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] p {
    font-family: var(--mono) !important; font-size: 10px !important;
    letter-spacing: 0.15em !important; color: var(--muted) !important;
    text-transform: uppercase !important;
}

div[data-testid="stAlert"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 7px !important;
    font-family: var(--mono) !important; font-size: 11px !important;
}
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 0 !important; }
.dash-rule { border: none; border-top: 1px dashed var(--border-hi); margin: 16px 0; }
::-webkit-scrollbar { width: 3px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ============================
# 5. BACKEND
# ============================
DB_PATH   = "data/face_db.pkl"
TEMP_PATH = "uploads/temp.jpg"
PERM_PATH = "uploads/perm/"

os.makedirs("data",    exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs(PERM_PATH, exist_ok=True)


@st.cache_resource
def load_models():
    detector = MTCNN()
    model = VGGFace(model="resnet50", include_top=False,
                    input_shape=(224, 224, 3), pooling="avg")
    return detector, model

detector, model = load_models()

if not os.path.exists(DB_PATH):
    with open(DB_PATH, "wb") as f:
        pickle.dump({}, f)
with open(DB_PATH, "rb") as f:
    face_db = pickle.load(f)


def save_db(db):
    with open(DB_PATH, "wb") as f: pickle.dump(db, f)


def extract_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces   = detector.detect_faces(img_rgb)
    if not faces: return None, None
    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)
    face_pil = Image.fromarray(img_rgb[y:y+h, x:x+w]).resize((224, 224))
    arr = preprocess_input(np.expand_dims(np.asarray(face_pil).astype("float32"), 0))
    return model.predict(arr, verbose=0).flatten(), face_pil


def predict_identity(embedding):
    best_name, best_score, best_img_path = "Unknown", 0.0, None
    for name, records in face_db.items():
        for r in records:
            sim = cosine_similarity(
                embedding.reshape(1,-1), r["embedding"].reshape(1,-1)
            )[0][0]
            if sim > best_score:
                best_score, best_name, best_img_path = sim, name, r["img_path"]
    return best_name, best_score, best_img_path


def save_identity(name, emb, current_path):
    ext = current_path.split(".")[-1]
    perm_path = os.path.join(PERM_PATH, f"{name}_{uuid.uuid4().hex[:6]}.{ext}")
    shutil.copy(current_path, perm_path)
    if name not in face_db: face_db[name] = []
    face_db[name].append({"embedding": emb, "img_path": perm_path})
    save_db(face_db)


# ============================
# 6. EVALUATION FUNCTION (table-based)
# ============================

def build_cm_html(cm, labels):
    """Render confusion matrix as a styled HTML table."""
    max_val = cm.max() if cm.max() > 0 else 1
    header_cells = '<th class="row-header">Actual \\ Predicted</th>'
    for lbl in labels:
        short = lbl[:10] + ("…" if len(lbl) > 10 else "")
        header_cells += f'<th>{short}</th>'

    rows = ""
    for i, actual in enumerate(labels):
        short_actual = actual[:12] + ("…" if len(actual) > 12 else "")
        row = f'<td class="label-cell">{short_actual}</td>'
        for j, _ in enumerate(labels):
            val  = int(cm[i][j])
            diag = "cm-diagonal" if i == j else ""
            ratio = val / max_val
            if val == 0:
                css = "cm-cell-zero"
                display = "—"
            elif ratio > 0.6:
                css = "cm-cell-high"
                display = str(val)
            elif ratio > 0.2:
                css = "cm-cell-med"
                display = str(val)
            else:
                css = ""
                display = str(val)
            row += f'<td class="{diag} {css}">{display}</td>'
        rows += f"<tr>{row}</tr>"

    return f"""
    <div class="cm-wrap">
        <div class="eval-table-header">
            <span class="eval-table-title">Confusion Matrix</span>
            <span class="eval-table-badge">{len(labels)} CLASSES</span>
        </div>
        <div class="cm-inner">
            <table class="cm-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>
    """


def evaluate_model(face_db):
    MAX_SAMPLES = 8000

    X, y = [], []
    for name, records in face_db.items():
        for r in records:
            if len(X) >= MAX_SAMPLES:
                break
            X.append(r["embedding"])
            y.append(name)

    if len(X) < 10:
        st.warning("Not enough data in database to run evaluation.")
        return

    n_total = len(X)
    n_classes = len(set(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    all_embeddings = np.array(X)
    all_labels     = np.array(y)
    y_pred         = []

    progress_bar = st.progress(0)
    total = len(X_test)
    for i, emb in enumerate(X_test):
        sims = cosine_similarity([emb], all_embeddings)[0]
        idx  = np.argmax(sims)
        y_pred.append(all_labels[idx])
        progress_bar.progress((i + 1) / total)

    acc = accuracy_score(y_test, y_pred)
    acc_pct = int(acc * 100)

    report = classification_report(y_test, y_pred, output_dict=True)

    labels_present = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)

    # ── meta chips ──────────────────────────────────────
    st.markdown(f"""
    <div class="eval-meta-row">
        <div class="eval-meta-chip">Samples used <span>{n_total}</span></div>
        <div class="eval-meta-chip">Test set <span>{len(X_test)}</span></div>
        <div class="eval-meta-chip">Train set <span>{len(X_train)}</span></div>
        <div class="eval-meta-chip">Classes <span>{n_classes}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── accuracy banner ──────────────────────────────────
    st.markdown(f"""
    <div class="acc-banner">
        <div>
            <div class="acc-banner-label">Overall Accuracy</div>
            <div class="acc-banner-value">{acc_pct}%</div>
            <div class="acc-banner-sub">cosine similarity · nearest-neighbor</div>
        </div>
        <div class="acc-track">
            <div class="acc-fill" style="width:{acc_pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── per-class stat tiles (up to 4 classes shown) ─────
    display_classes = [c for c in labels_present if c in report and c not in ("accuracy","macro avg","weighted avg")][:4]
    if display_classes:
        tiles_html = '<div class="eval-grid">'
        colors = ["#7b6cff","#00cfb4","#f0a500","#ff4f4f"]
        for idx2, cls in enumerate(display_classes):
            p  = report[cls]
            f1 = int(p["f1-score"] * 100)
            c  = colors[idx2 % len(colors)]
            short_cls = cls[:14] + ("…" if len(cls) > 14 else "")
            tiles_html += f"""
            <div class="eval-tile">
                <div class="eval-tile-label">{short_cls}</div>
                <div class="eval-tile-val">{f1}<span style="font-size:13px;color:var(--muted)">%</span></div>
                <div class="eval-tile-sub">f1-score · support {int(p['support'])}</div>
                <div class="eval-tile-bar">
                    <div class="eval-tile-bar-fill" style="width:{f1}%;background:{c}"></div>
                </div>
            </div>"""
        tiles_html += "</div>"
        st.markdown(tiles_html, unsafe_allow_html=True)

    # ── classification report table ───────────────────────
    report_rows = []
    for cls in labels_present:
        if cls in report:
            p = report[cls]
            report_rows.append({
                "Class":     cls,
                "Precision": f"{p['precision']:.3f}",
                "Recall":    f"{p['recall']:.3f}",
                "F1-Score":  f"{p['f1-score']:.3f}",
                "Support":   int(p["support"]),
            })
    for avg in ["macro avg", "weighted avg"]:
        if avg in report:
            p = report[avg]
            report_rows.append({
                "Class":     avg,
                "Precision": f"{p['precision']:.3f}",
                "Recall":    f"{p['recall']:.3f}",
                "F1-Score":  f"{p['f1-score']:.3f}",
                "Support":   int(p["support"]),
            })

    st.markdown("""
    <div class="eval-table-wrap">
        <div class="eval-table-header">
            <span class="eval-table-title">Classification Report</span>
            <span class="eval-table-badge">PER-CLASS METRICS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    df_report = pd.DataFrame(report_rows).set_index("Class")
    st.dataframe(df_report, use_container_width=True)

    # ── confusion matrix (HTML table) ────────────────────
    st.markdown(build_cm_html(cm, labels_present), unsafe_allow_html=True)


# ============================
# 7. UI — MASTHEAD
# ============================
st.markdown("""
<div class="masthead">
    <div class="masthead-logo"><span>◈</span> FaceID Pro</div>
    <div class="masthead-tag">BIOMETRIC SCANNER v2.0</div>
</div>
""", unsafe_allow_html=True)

# ============================
# 8. UI — RECOGNITION PANEL
# ============================
st.markdown('<p class="upload-eyebrow">Subject Image Input</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "png"],
                             key="main_uploader", label_visibility="collapsed")

if not uploaded:
    st.info("Awaiting subject image — drag & drop or click to upload.")
else:
    with open(TEMP_PATH, "wb") as f:
        f.write(uploaded.getbuffer())

    emb, face_crop = extract_embedding(TEMP_PATH)
    if emb is None:
        st.error("No face detected in the uploaded image.")
    else:
        name, score, match_img_path = predict_identity(emb)
        score_pct = int(score * 100)

        if score > 0.75:
            dyn_color       = "#00e5a0"
            dyn_chip_bg     = "rgba(0,229,160,0.08)"
            dyn_chip_border = "rgba(0,229,160,0.22)"
            status_text     = "VERIFIED"
        elif score > 0.60:
            dyn_color       = "#f0a500"
            dyn_chip_bg     = "rgba(240,165,0,0.08)"
            dyn_chip_border = "rgba(240,165,0,0.22)"
            status_text     = "UNCERTAIN"
        else:
            dyn_color       = "#ff4f4f"
            dyn_chip_bg     = "rgba(255,79,79,0.08)"
            dyn_chip_border = "rgba(255,79,79,0.22)"
            status_text     = "UNKNOWN"

        st.markdown(f"""
        <style>
        :root {{
            --dyn-color:       {dyn_color};
            --dyn-pct:         {score_pct}%;
            --dyn-chip-bg:     {dyn_chip_bg};
            --dyn-chip-border: {dyn_chip_border};
        }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        col_vis, col_ctrl = st.columns([3, 2], gap="large")

        with col_vis:
            st.markdown('<p class="sec-label">Visual Analysis</p>', unsafe_allow_html=True)
            img_col1, img_col2 = st.columns(2, gap="small")

            with img_col1:
                st.markdown("""
                <div class="img-frame">
                  <div class="img-frame-header">
                    <span class="img-frame-title">Incoming Subject</span>
                    <span class="img-frame-dot" style="background:#7b6cff"></span>
                  </div>
                  <div class="img-frame-body">
                """, unsafe_allow_html=True)
                st.image(uploaded, use_container_width=False)
                st.markdown("</div></div>", unsafe_allow_html=True)

            with img_col2:
                st.markdown("""
                <div class="img-frame">
                  <div class="img-frame-header">
                    <span class="img-frame-title">DB Reference</span>
                    <span class="img-frame-dot" style="background:#00cfb4"></span>
                  </div>
                  <div class="img-frame-body">
                """, unsafe_allow_html=True)
                if match_img_path:
                    st.image(Image.open(match_img_path), use_container_width=False)
                else:
                    st.markdown(
                        '<p style="font-family:var(--mono);font-size:11px;'
                        'color:var(--muted);padding:28px;text-align:center;">'
                        'No reference in database</p>',
                        unsafe_allow_html=True
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)

        with col_ctrl:
            st.markdown('<p class="sec-label">Identification Result</p>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card">
                <div class="rc-header">
                    <div class="rc-header-dot"></div>
                    <span class="rc-header-label">Identity</span>
                </div>
                <div class="rc-body">
                    <p class="id-name">{name}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="sec-label" style="margin-top:18px">Action</p>',
                        unsafe_allow_html=True)

            b1, b2 = st.columns(2, gap="small")
            with b1:
                if st.button("✓  Confirm", use_container_width=True):
                    save_identity(name, emb, TEMP_PATH)
                    st.success(f"Confirmed: {name}")
                    st.balloons()
            with b2:
                if st.button("✕  Reject", use_container_width=True):
                    st.session_state["show_correction"] = True

            st.markdown('<hr class="dash-rule"/>', unsafe_allow_html=True)
            st.markdown('<p class="sec-label">Manual Override</p>', unsafe_allow_html=True)

            with st.form("fix_identity"):
                new_name  = st.text_input("Correct Name", placeholder="Enter identity…")
                submitted = st.form_submit_button("Update Database", use_container_width=True)
                if submitted and new_name:
                    save_identity(new_name, emb, TEMP_PATH)
                    st.success(f"Database updated → {new_name}")
                    st.session_state["show_correction"] = False
                    st.rerun()

# ============================
# 9. UI — EVALUATION PANEL
# ============================
st.markdown("""
<div class="eval-section-title">Model Evaluation</div>
""", unsafe_allow_html=True)

if st.button("⬡  Run Evaluation", use_container_width=False):
    evaluate_model(face_db)