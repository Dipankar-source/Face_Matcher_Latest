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

if not hasattr(st, 'rerun'):
    st.rerun = st.experimental_rerun

# ============================
# 3. CONFIG
# ============================
st.set_page_config(layout="wide", page_title="MukhPehchaan", page_icon="👁️")

DB_PATH = "data/face_db.pkl"
TEMP_PATH = "uploads/temp.jpg"
PERM_PATH = "uploads/perm/"

os.makedirs("data", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs(PERM_PATH, exist_ok=True)

# ============================
# 4. LOAD MODELS
# ============================
@st.cache_resource
def load_models():
    detector = MTCNN()
    model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg")
    return detector, model

detector, model = load_models()

# ============================
# 5. DB INIT
# ============================
if not os.path.exists(DB_PATH):
    with open(DB_PATH, "wb") as f:
        pickle.dump({}, f)

with open(DB_PATH, "rb") as f:
    face_db = pickle.load(f)

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

# ============================
# 6. CORE FUNCTIONS
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
    best_name, best_score, best_img_path = "Unknown", 0.0, None

    for name, records in face_db.items():
        for r in records:
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                r["embedding"].reshape(1, -1)
            )[0][0]

            if sim > best_score:
                best_score = sim
                best_name = name
                best_img_path = r.get("img_path", None)

    return best_name, best_score, best_img_path


def save_identity(name, emb, current_path):
    ext = current_path.split('.')[-1]
    perm_path = os.path.join(PERM_PATH, f"{name}_{uuid.uuid4().hex[:6]}.{ext}")

    try:
        shutil.copy(current_path, perm_path)
    except Exception:
        return  # fail silently if copy fails (cloud safe)

    if name not in face_db:
        face_db[name] = []

    face_db[name].append({
        "embedding": emb,
        "img_path": perm_path
    })

    save_db(face_db)

# ============================
# 7. UI
# ============================

st.title("👁️ MukhPehchaan - Face Recognition System")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

if not uploaded:
    st.info("Upload an image to begin")
    st.stop()

with open(TEMP_PATH, "wb") as f:
    f.write(uploaded.getbuffer())

emb, face_crop = extract_embedding(TEMP_PATH)

if emb is None:
    st.error("No face detected")
    st.stop()

name, score, match_img_path = predict_identity(emb)

col1, col2 = st.columns(2)

# LEFT: INPUT IMAGE
with col1:
    st.subheader("Uploaded Image")
    st.image(uploaded, use_container_width=True)

# RIGHT: MATCH IMAGE (FIXED HERE)
with col2:
    st.subheader("Matched Image")

    if match_img_path and os.path.exists(match_img_path):
        try:
            ref_img = Image.open(match_img_path)
            st.image(ref_img, use_container_width=True)
        except Exception:
            st.warning("Image exists but cannot be opened")
    else:
        st.warning("No reference image found")

# ============================
# 8. RESULT
# ============================

st.markdown(f"### 🧠 Prediction: {name}")
st.markdown(f"### 📊 Confidence: {score:.2f}")

# ============================
# 9. ACTIONS
# ============================

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Confirm"):
        save_identity(name, emb, TEMP_PATH)
        st.success("Saved")

with col2:
    new_name = st.text_input("Enter correct name")

    if st.button("Update") and new_name:
        save_identity(new_name, emb, TEMP_PATH)
        st.success("Updated")
