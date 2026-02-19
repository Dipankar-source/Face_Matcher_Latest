# ============================
# KERAS-VGGFACE COMPAT PATCH
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
import numpy as np
import pickle
import cv2
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

# ============================
# LOAD DATA
# ============================
feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# ============================
# LOAD MODEL
# ============================
model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg'
)

# ============================
# FACE DETECTION
# ============================
detector = MTCNN()

sample_img = cv2.imread('sample/rashmika.png')

# BGR ➜ RGB (IMPORTANT)
rgb_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

results = detector.detect_faces(rgb_img)

if len(results) == 0:
    raise Exception("No face detected in image")

x, y, width, height = results[0]['box']

# Fix negative coordinates
x, y = max(0, x), max(0, y)

face = rgb_img[y:y+height, x:x+width]

# ============================
# FEATURE EXTRACTION
# ============================
image = Image.fromarray(face).resize((224,224))
face_array = np.asarray(image).astype('float32')

expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)

result = model.predict(preprocessed_img, verbose=0).flatten()

# ============================
# COSINE SIMILARITY
# ============================
similarity = cosine_similarity(
    result.reshape(1, -1),
    feature_list
)[0]

index_pos = np.argmax(similarity)

# ============================
# SHOW RESULT
# ============================
matched_img = cv2.imread(filenames[index_pos])

cv2.imshow('Matched Face', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
