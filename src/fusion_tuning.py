"""
Offline tuning script for image+clinical fusion.

Usage:
  - Assumes you have:
      models/dense_best.h5
      models/clinical_best.pkl
      models/clinical_best_scaler.pkl
      data/chest_xray/test with NORMAL / PNEUMONIA folders
      data/clinical/clinical_data.csv
  - Computes simulated validation predictions
  - Searches over fusion weights and thresholds using ROC‑AUC / F1
  - Writes best configuration to models/fusion_config.json

This is meant for offline experimentation and not imported by the Flask app at runtime.
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

IMAGE_MODEL_PATH = "models/dense_best.h5"
CLINICAL_MODEL_PATH = "models/clinical_best.pkl"
SCALER_PATH = "models/clinical_best_scaler.pkl"
CLINICAL_CSV = "data/clinical/clinical_data.csv"
TEST_IMAGE_DIR = "data/chest_xray/test"


def load_models():
    import pickle

    print("Loading image model...")
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    print("Loading clinical model and scaler...")
    with open(CLINICAL_MODEL_PATH, "rb") as f:
        clinical_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return image_model, clinical_model, scaler


def collect_predictions():
    image_model, clinical_model, scaler = load_models()
    df_clin = pd.read_csv(CLINCIAL_CSV) if os.path.exists(CLINCIAL_CSV := CLINICAL_CSV) else None

    feats = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]
    all_y = []
    img_probs = []
    clin_probs = []

    classes = {"NORMAL": 0, "PNEUMONIA": 1}
    for cls_name, label in classes.items():
        folder = os.path.join(TEST_IMAGE_DIR, cls_name)
        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            ip = float(image_model.predict(arr, verbose=0)[0][0])

            if df_clin is not None:
                row = df_clin.sample(1)[feats]
                row_scaled = scaler.transform(row)
                cp = float(clinical_model.predict_proba(row_scaled)[0][1])
            else:
                cp = 0.5

            all_y.append(label)
            img_probs.append(ip)
            clin_probs.append(cp)

    return np.array(all_y), np.array(img_probs), np.array(clin_probs)


def search_fusion(y_true, img_p, clin_p):
    best = {"f1": -1.0}
    for w in np.linspace(0.5, 0.95, 10):
        fusion = w * img_p + (1 - w) * clin_p
        for thr in np.linspace(0.3, 0.7, 9):
            preds = (fusion >= thr).astype(int)
            f1 = f1_score(y_true, preds)
            auc = roc_auc_score(y_true, fusion)
            acc = accuracy_score(y_true, preds)
            if f1 > best["f1"]:
                best = {
                    "w_image": float(w),
                    "w_clinical": float(1 - w),
                    "threshold": float(thr),
                    "f1": float(f1),
                    "roc_auc": float(auc),
                    "accuracy": float(acc),
                }
    return best


def main():
    print("Collecting simulated validation predictions...")
    y_true, img_p, clin_p = collect_predictions()
    print(f"Collected {len(y_true)} samples.")

    print("Searching fusion weights and threshold...")
    best = search_fusion(y_true, img_p, clin_p)

    print("\n=== BEST FUSION CONFIG (SIMULATED) ===")
    for k, v in best.items():
        print(f"{k}: {v}")

    os.makedirs("models", exist_ok=True)
    with open("models/fusion_config.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nSaved fusion configuration to models/fusion_config.json")
    print("Label: SIMULATED EVALUATION – use for research/demo only.")


if __name__ == "__main__":
    main()


