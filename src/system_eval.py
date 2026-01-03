"""
System‑level evaluation script for the AI Healthcare Diagnosis project.

This script performs a SIMULATED EVALUATION of the end‑to‑end system using:
  - Image model (dense_best.h5)
  - Clinical model (clinical_best.pkl)
  - Weighted fusion (optionally from fusion_config.json)

Metrics reported:
  - Confusion matrix
  - Accuracy, Precision, Recall, F1
  - ROC‑AUC

IMPORTANT: Results are labelled as simulated because clinical records are
randomly paired with test images. Use only for research/demo reporting.
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_components():
    import pickle

    image_model = tf.keras.models.load_model("models/dense_best.h5")
    with open("models/clinical_best.pkl", "rb") as f:
        clinical_model = pickle.load(f)
    with open("models/clinical_best_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    w_img, w_clin, thr = 0.85, 0.15, 0.5
    if os.path.exists("models/fusion_config.json"):
        with open("models/fusion_config.json", "r") as f:
            cfg = json.load(f)
        w_img = cfg.get("w_image", w_img)
        w_clin = cfg.get("w_clinical", 1.0 - w_img)
        thr = cfg.get("threshold", thr)
    return image_model, clinical_model, scaler, w_img, w_clin, thr


def main():
    image_model, clinical_model, scaler, w_img, w_clin, thr = load_components()

    df_clin = pd.read_csv("data/clinical/clinical_data.csv")
    feats = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]

    IMAGE_DIR = "data/chest_xray/test"
    CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}

    y_true = []
    img_p = []
    clin_p = []

    for class_name, label in CLASSES.items():
        folder = os.path.join(IMAGE_DIR, class_name)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            ip = float(image_model.predict(arr, verbose=0)[0][0])

            clin_row = df_clin.sample(1)[feats]
            clin_scaled = scaler.transform(clin_row)
            cp = float(clinical_model.predict_proba(clin_scaled)[0][1])

            y_true.append(label)
            img_p.append(ip)
            clin_p.append(cp)

    y_true = np.array(y_true)
    img_p = np.array(img_p)
    clin_p = np.array(clin_p)

    fusion = w_img * img_p + w_clin * clin_p
    y_pred = (fusion >= thr).astype(int)

    print("\n=== SIMULATED SYSTEM EVALUATION ===")
    print(f"Fusion weights: image={w_img:.2f}, clinical={w_clin:.2f}, threshold={thr:.2f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

    print("\nScalar metrics:")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC‑AUC  : {roc_auc_score(y_true, fusion):.4f}")

    print("\nLabel: SIMULATED EVALUATION – do not interpret as real‑world performance without proper pairing.")


if __name__ == "__main__":
    main()


