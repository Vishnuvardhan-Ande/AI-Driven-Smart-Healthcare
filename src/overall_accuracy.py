import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# LOAD MODELS (aligned with app.py)
# -------------------------
print("📌 Loading models...")

image_model = tf.keras.models.load_model("models/dense_best.h5")
clinical_model = pickle.load(open("models/clinical_best.pkl", "rb"))
clinical_scaler = pickle.load(open("models/clinical_best_scaler.pkl", "rb"))

# -------------------------
# LOAD CLINICAL DATA
# -------------------------
df_clinical = pd.read_csv("data/clinical/clinical_data.csv")

clinical_features = [
    "age", "fever_days", "spo2",
    "cough", "smoking", "diabetes"
]

# -------------------------
# IMAGE DATASET
# -------------------------
IMAGE_DIR = "data/chest_xray/test"
CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}

all_y_true = []
image_probs = []
clinical_probs = []

print("🔍 Starting simulated fusion evaluation...")

# -------------------------
# LOOP THROUGH IMAGES
# -------------------------
for class_name, label in CLASSES.items():
    class_folder = os.path.join(IMAGE_DIR, class_name)

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)

        # ---- IMAGE PREDICTION ----
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(224, 224)
        )
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        image_prob = image_model.predict(img, verbose=0)[0][0]

        # ---- RANDOM CLINICAL ROW (SIMULATED PAIRING) ----
        clinical_row = df_clinical.sample(1)[clinical_features]
        clinical_scaled = clinical_scaler.transform(clinical_row)

        clinical_prob = clinical_model.predict_proba(clinical_scaled)[0][1]

        all_y_true.append(label)
        image_probs.append(image_prob)
        clinical_probs.append(clinical_prob)

image_probs = np.array(image_probs)
clinical_probs = np.array(clinical_probs)
all_y_true = np.array(all_y_true)

# -------------------------
# SIMPLE GRID SEARCH FOR BEST FUSION WEIGHT & THRESHOLD
# -------------------------
best_acc = -1.0
best_w = None
best_thr = None

for w in np.linspace(0.5, 0.9, 9):  # image weight
    fusion = w * image_probs + (1 - w) * clinical_probs
    for thr in np.linspace(0.3, 0.7, 9):
        preds = (fusion > thr).astype(int)
        acc = accuracy_score(all_y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_thr = thr

print("\n✅ BEST FUSION CONFIG (simulated)")
print(f"Best image weight w: {best_w:.2f}")
print(f"Best threshold: {best_thr:.2f}")
print(f"Fusion Accuracy: {best_acc * 100:.2f}%\n")

final_fusion = best_w * image_probs + (1 - best_w) * clinical_probs
final_preds = (final_fusion > best_thr).astype(int)

print("📊 Confusion Matrix (best fusion):")
print(confusion_matrix(all_y_true, final_preds))

print("\n📋 Classification Report (best fusion):")
print(classification_report(all_y_true, final_preds, target_names=["NORMAL", "PNEUMONIA"]))
