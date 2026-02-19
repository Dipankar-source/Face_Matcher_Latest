# ============================
# KERAS-VGGFACE PATCH
# ============================
import sys, types
from tensorflow.keras.utils import get_source_inputs

keras_engine = types.ModuleType("keras.engine")
keras_engine_topology = types.ModuleType("keras.engine.topology")
keras_engine_topology.get_source_inputs = get_source_inputs

sys.modules["keras.engine"] = keras_engine
sys.modules["keras.engine.topology"] = keras_engine_topology

# ============================
# IMPORTS
# ============================
import os
import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from tqdm import tqdm

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# ============================
# CONFIG
# ============================
DATASET_DIR = "dataset"
OUTPUT_DB = "data/face_db.pkl"

os.makedirs("data", exist_ok=True)

# ============================
# LOAD MODELS
# ============================
detector = MTCNN()
model = VGGFace(
    model="resnet50",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)

# ============================
# FUNCTIONS
# ============================
def extract_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if not faces:
        return None

    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)

    face = img_rgb[y:y+h, x:x+w]
    face = Image.fromarray(face).resize((224, 224))

    arr = np.asarray(face).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    return model.predict(arr, verbose=0).flatten()

# ============================
# BUILD FACE DATABASE
# ============================
face_db = {}

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    face_db[person] = []

    for img_name in tqdm(os.listdir(person_path), desc=f"Processing {person}"):
        img_path = os.path.join(person_path, img_name)
        emb = extract_embedding(img_path)

        if emb is not None:
            face_db[person].append({
                "embedding": emb,
                "img_path": img_path
            })

# ============================
# SAVE DATABASE
# ============================
with open(OUTPUT_DB, "wb") as f:
    pickle.dump(face_db, f)

print("✅ face_db.pkl created successfully")
