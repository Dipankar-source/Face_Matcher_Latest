import os
import pickle

DATASET_DIR = "dataset"   # change if needed

filenames = []

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, file)
            filenames.append(full_path)

print(f"Total images found: {len(filenames)}")

with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print("filenames.pkl created successfully")
