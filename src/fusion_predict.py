import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import cv2
import os

def load_models():
    image_model = tf.keras.models.load_model("models/dense_best.h5")
    clinical_model = joblib.load("models/clinical_rf.pkl")
    return image_model, clinical_model

def preprocess_image(img_path):
    IMG_SIZE = (224, 224)
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_fusion(img_path, clinical_row):
    image_model, clinical_model = load_models()

    # IMAGE prediction
    img = preprocess_image(img_path)
    image_prob = float(image_model.predict(img, verbose=0)[0][0])

    # CLINICAL prediction
    clinical_input = np.array(clinical_row).reshape(1, -1)
    clinical_prob = float(clinical_model.predict_proba(clinical_input)[0][1])

    # FUSION prediction
    final_score = (0.6 * image_prob) + (0.4 * clinical_prob)

    final_label = 1 if final_score >= 0.5 else 0

    return {
        "image_prob": round(image_prob, 4),
        "clinical_prob": round(clinical_prob, 4),
        "final_score": round(final_score, 4),
        "final_label": final_label
    }

if __name__ == "__main__":
    example_img = "test_images/ti1.jpg"

    example_clinical = [55, 5, 89, 1, 1, 1]  

    out = predict_fusion(example_img, example_clinical)
    print(out)
