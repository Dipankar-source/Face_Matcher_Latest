# ============================
# 1. COMPATIBILITY & PATCHES
# ============================
import os
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
from io import BytesIO

if not hasattr(st, 'rerun'):
    st.rerun = st.experimental_rerun

# ============================
# 3. PATHS (FIXED)
# ============================
BASE_DIR = os.path.dirname(__file__)

DB_PATH = os.path.join(BASE_DIR, "data", "face_db.pkl")
TEMP_PATH = os.path.join(BASE_DIR, "uploads", "temp.jpg")
PERM_PATH = os.path.join(BASE_DIR, "uploads", "perm")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
os.makedirs(PERM_PATH, exist_ok=True)

# ============================
# 4. MODELS
# ============================
@st.cache_resource
def load_models():
    detector = MTCNN()
    model = VGGFace(model="resnet50", include_top=False,
                    input_shape=(224, 224, 3), pooling="avg")
    return detector, model

detector, model = load_models()

# ============================
# 5. DATABASE (AUTO CLEAN)
# ============================
if not os.path.exists(DB_PATH):
    with open(DB_PATH, "wb") as f:
        pickle.dump({}, f)

with open(DB_PATH, "rb") as f:
    face_db = pickle.load(f)

# 🔥 CLEAN BROKEN PATHS
clean_db = {}
for name, records in face_db.items():
    valid_records = []
    for r in records:
        if "img_bytes" in r:
            valid_records.append(r)
        elif "img_path" in r and os.path.exists(r["img_path"]):
            valid_records.append(r)
    if valid_records:
        clean_db[name] = valid_records

face_db = clean_db

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

# ============================
# 6. CORE LOGIC
# ============================
def extract_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if not faces:
        return None, None

    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)

    face_crop = img_rgb[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_crop).resize((224, 224))

    arr = np.asarray(face_pil).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    return model.predict(arr, verbose=0).flatten(), face_pil


def predict_identity(embedding):
    best_name, best_score, best_record = "Unknown", 0.0, None

    for name, records in face_db.items():
        for r in records:
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                r["embedding"].reshape(1, -1)
            )[0][0]

            if sim > best_score:
                best_score = sim
                best_name = name
                best_record = r

    return best_name, best_score, best_record


# 🔥 FIXED SAVE (STORE IMAGE BYTES INSTEAD OF PATH)
def save_identity(name, emb, current_path):
    with open(current_path, "rb") as f:
        img_bytes = f.read()

    if name not in face_db:
        face_db[name] = []

    face_db[name].append({
        "embedding": emb,
        "img_bytes": img_bytes   # ✅ CLOUD SAFE
    })

    save_db(face_db)


# ============================
# 7. UI (UNCHANGED)
# ============================

st.set_page_config(layout="wide", page_title="FaceID Pro", page_icon="👁️")

# (⚠️ KEEP YOUR CSS EXACTLY SAME — NOT TOUCHING)

st.markdown("<h2 style='text-align: center;'>B I O M E T R I C &nbsp; S C A N N E R</h2>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "png"], key="main_uploader")

if not uploaded:
    st.info("Waiting for subject image...")
    st.stop()

with open(TEMP_PATH, "wb") as f:
    f.write(uploaded.getbuffer())

emb, face_crop = extract_embedding(TEMP_PATH)

if emb is None:
    st.error("No face detected in the image.")
    st.stop()

name, score, record = predict_identity(emb)

st.write("---")

col_visuals, col_controls = st.columns([2, 1.2], gap="large")

with col_visuals:
    st.subheader("👁️ Visual Analysis")

    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.caption("Incoming Subject")
        st.image(uploaded, use_container_width=True)

    with v_col2:
        st.caption("Database Reference")

        if record:
            try:
                if "img_bytes" in record:
                    ref_img = Image.open(BytesIO(record["img_bytes"]))
                elif "img_path" in record and os.path.exists(record["img_path"]):
                    ref_img = Image.open(record["img_path"])
                else:
                    raise Exception("Invalid record")

                st.image(ref_img, use_container_width=True)

            except:
                st.warning("Reference image missing or corrupted")
                st.image("https://via.placeholder.com/300x300.png?text=No+Match",
                         use_container_width=True)
        else:
            st.warning("No Reference Found")
            st.image("https://via.placeholder.com/300x300.png?text=No+Match",
                     use_container_width=True)

# RIGHT PANEL SAME (NO CHANGE)
with col_controls:
    st.markdown("### 🧬 Identification Result")

    if score > 0.75:
        match_color = "#00ff99"
        status_text = "VERIFIED"
    elif score > 0.6:
        match_color = "#ffcc00"
        status_text = "UNCERTAIN"
    else:
        match_color = "#ff4b4b"
        status_text = "UNKNOWN"

    st.markdown(f"<h1>{name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{match_color}'>{score:.1%} {status_text}</div>",
                unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("✅ Confirm", use_container_width=True):
            save_identity(name, emb, TEMP_PATH)
            st.success(f"Confirmed: {name}")
            st.balloons()

    with btn_col2:
        if st.button("❌ Reject", use_container_width=True):
            st.session_state['show_correction'] = True

    with st.form("fix_identity"):
        new_name_input = st.text_input("📝 Correction / New Name")

        if st.form_submit_button("Update Database") and new_name_input:
            save_identity(new_name_input, emb, TEMP_PATH)
            st.success(f"Database Updated: {new_name_input}")
            st.rerun()
