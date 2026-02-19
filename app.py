# ============================
# 1. COMPATIBILITY & PATCHES
# ============================
import os
# Fix for "Descriptors cannot be created directly" (Protobuf error)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys, types
from tensorflow.keras.utils import get_source_inputs

keras_engine = types.ModuleType("keras.engine")
keras_engine_topology = types.ModuleType("keras.engine.topology")
keras_engine_topology.get_source_inputs = get_source_inputs

sys.modules["keras.engine"] = keras_engine
sys.modules["keras.engine.topology"] = keras_engine_topology

# ============================
# 2. IMPORTS
# ============================
import cv2
import pickle
import numpy as np
import shutil
import uuid
import streamlit as st
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Fix for old Streamlit versions
if not hasattr(st, 'rerun'):
    st.rerun = st.experimental_rerun

# ============================
# 3. PAGE CONFIG & MODERN CSS
# ============================
st.set_page_config(layout="wide", page_title="FaceID Pro", page_icon="👁️")

st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp {
        background: linear-gradient(135deg, #1e2029 0%, #000000 100%);
        color: #FFFFFF;
    }

    /* CENTERED UPLOADER */
    div[data-testid="stFileUploader"] {
        width: 50%;
        margin: 0 auto;
        padding-top: 20px;
    }
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #2b2e3b;
        border: 2px dashed #4e5d6c;
        border-radius: 20px;
    }

    /* GLASSMORPHISM CARD (For Right Panel) */
    .control-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        height: 100%;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* IMAGE CONTAINERS */
    .img-container {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid #333;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: transform 0.3s ease;
    }
    .img-container:hover {
        transform: scale(1.02);
    }
    
    /* TILTED SEPARATOR */
    .tilted-divider {
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #ff00cc, #333399);
        transform: skewY(-2deg);
        margin: 20px 0;
        border-radius: 2px;
    }

    /* METRICS */
    .metric-box {
        font-size: 2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* INPUT FIELD STYLING */
    .stTextInput input {
        background-color: #1a1c24;
        color: white;
        border-radius: 10px;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# 4. BACKEND LOGIC
# ============================
DB_PATH = "data/face_db.pkl"
TEMP_PATH = "uploads/temp.jpg"
PERM_PATH = "uploads/perm/"

os.makedirs("data", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs(PERM_PATH, exist_ok=True)

@st.cache_resource
def load_models():
    detector = MTCNN()
    model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")
    return detector, model

detector, model = load_models()

if not os.path.exists(DB_PATH):
    with open(DB_PATH, "wb") as f: pickle.dump({}, f)

with open(DB_PATH, "rb") as f:
    face_db = pickle.load(f)

def save_db(db):
    with open(DB_PATH, "wb") as f: pickle.dump(db, f)

def extract_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    if not faces: return None, None
    
    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)
    
    face_crop = img_rgb[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_crop).resize((224, 224))
    
    arr = np.asarray(face_pil).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    return model.predict(arr, verbose=0).flatten(), face_pil

def predict_identity(embedding):
    best_name, best_score, best_img_path = "Unknown", 0.0, None
    if not face_db: return best_name, best_score, best_img_path
    
    for name, records in face_db.items():
        for r in records:
            sim = cosine_similarity(embedding.reshape(1, -1), r["embedding"].reshape(1, -1))[0][0]
            if sim > best_score:
                best_score = sim
                best_name = name
                best_img_path = r["img_path"]
    return best_name, best_score, best_img_path

def save_identity(name, emb, current_path):
    ext = current_path.split('.')[-1]
    perm_path = os.path.join(PERM_PATH, f"{name}_{uuid.uuid4().hex[:6]}.{ext}")
    shutil.copy(current_path, perm_path)
    
    if name not in face_db: face_db[name] = []
    face_db[name].append({"embedding": emb, "img_path": perm_path})
    save_db(face_db)

# ============================
# 5. UI IMPLEMENTATION
# ============================

# --- TOP: UPLOADER ---
st.markdown("<h2 style='text-align: center; font-weight: 300; letter-spacing: 2px;'>B I O M E T R I C &nbsp; S C A N N E R</h2>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "png"], key="main_uploader")

if not uploaded:
    st.info("Waiting for subject image...")
    st.stop()

# Process Image
with open(TEMP_PATH, "wb") as f:
    f.write(uploaded.getbuffer())

emb, face_crop = extract_embedding(TEMP_PATH)

if emb is None:
    st.error("No face detected in the image.")
    st.stop()

name, score, match_img_path = predict_identity(emb)

st.write("---")

# --- MAIN GRID: GOLDEN RATIO SPLIT ---
# Left (Visuals) = 62% | Right (Controls) = 38%
col_visuals, col_controls = st.columns([2, 1.2], gap="large")

# === LEFT PORTION: THE VISUALS ===
with col_visuals:
    st.subheader("👁️ Visual Analysis")
    
    # We split the left portion again for side-by-side images
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.caption("Incoming Subject")
        # UPDATED: Replaced use_column_width with use_container_width
        st.image(uploaded, use_container_width=True, channels="RGB")
        
    with v_col2:
        st.caption("Database Reference")
        if match_img_path:
            ref_img = Image.open(match_img_path)
            # UPDATED: Replaced use_column_width with use_container_width
            st.image(ref_img, use_container_width=True)
        else:
            st.warning("No Reference Found")
            # UPDATED: Replaced use_column_width with use_container_width
            st.image("https://via.placeholder.com/300x300.png?text=No+Match", use_container_width=True)

# === RIGHT PORTION: CONTROLS & DATA ===
with col_controls:
    # Start the "Glass" Card
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    st.markdown("### 🧬 Identification Result")
    
    # Stylish Separator
    st.markdown('<div class="tilted-divider"></div>', unsafe_allow_html=True)
    
    # Name & Score
    if score > 0.75:
        match_color = "#00ff99" # Green
        status_text = "VERIFIED"
    elif score > 0.6:
        match_color = "#ffcc00" # Yellow
        status_text = "UNCERTAIN"
    else:
        match_color = "#ff4b4b" # Red
        status_text = "UNKNOWN"

    st.markdown(f"<p style='color: #aaa; margin-bottom: 0;'>Identity:</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top: 0;'>{name}</h1>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='color: #aaa; margin-bottom: 0;'>Confidence:</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box' style='color:{match_color}'>{score:.1%} <span style='font-size: 1rem; color:white; border:1px solid {match_color}; padding: 2px 8px; border-radius: 4px;'>{status_text}</span></div>", unsafe_allow_html=True)

    st.markdown('<div class="tilted-divider"></div>', unsafe_allow_html=True)
    
    # --- ACTIONS ---
    st.markdown("#### Action Required")
    
    # Two columns for buttons
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        # LIKE BUTTON (CONFIRM)
        if st.button("✅ Confirm", use_container_width=True):
            save_identity(name, emb, TEMP_PATH)
            st.success(f"Confirmed: {name}")
            st.balloons()
            
    with btn_col2:
        # DISLIKE BUTTON (TRIGGER CORRECTION)
        if st.button("❌ Reject", use_container_width=True):
            st.session_state['show_correction'] = True

    # --- INPUT FIELD (Always there, but highlighted if rejected) ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # If rejected, or just manually correcting
    with st.form("fix_identity"):
        new_name_input = st.text_input("📝 Correction / New Name", placeholder="Enter correct name here...")
        
        # Submit Button
        apply_btn = st.form_submit_button("Update Database", type="primary")
        
        if apply_btn and new_name_input:
            save_identity(new_name_input, emb, TEMP_PATH)
            st.success(f"Database Updated: {new_name_input}")
            # Reset state
            st.session_state['show_correction'] = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True) # End Glass Card