import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# LOAD MODELS
# -------------------------
print("📌 Loading models...")

image_model = tf.keras.models.load_model("models/dense_best.h5")
clinical_model = pickle.load(open("models/clinical_rf.pkl", "rb"))
clinical_scaler = pickle.load(open("models/clinical_scaler.pkl", "rb"))

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

y_true = []
y_pred = []

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

        # ---- FUSION ----
        fusion_prob = (image_prob + clinical_prob) / 2
        fusion_label = 1 if fusion_prob > 0.5 else 0

        y_true.append(label)
        y_pred.append(fusion_label)

# -------------------------
# RESULTS
# -------------------------
accuracy = accuracy_score(y_true, y_pred) * 100

print("\n✅ SIMULATED FUSION ACCURACY")
print(f"Fusion Accuracy: {accuracy:.2f}%\n")

print("📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
