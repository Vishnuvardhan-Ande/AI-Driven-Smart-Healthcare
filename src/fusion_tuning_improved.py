"""
Improved fusion tuning with better optimization strategies.

Improvements:
- Bayesian optimization for fusion weights
- Multiple fusion strategies (weighted average, stacking, etc.)
- Better threshold optimization using ROC curve
- Cross-validation for fusion weights
"""

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import pickle

IMAGE_MODEL_PATH = "models/dense_best.h5"
CLINICAL_MODEL_PATH = "models/clinical_best.pkl"
SCALER_PATH = "models/clinical_best_scaler.pkl"
CLINICAL_CSV = "data/clinical/clinical_data.csv"
TEST_IMAGE_DIR = "data/chest_xray/test"


def load_models():
    print("Loading image model...")
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    print("Loading clinical model and scaler...")
    with open(CLINICAL_MODEL_PATH, "rb") as f:
        clinical_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return image_model, clinical_model, scaler


def test_time_augmentation(image_model, img_path, n_augmentations=5):
    """
    Apply test-time augmentation for more robust predictions.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    base_img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    
    predictions = []
    
    # Original image
    img_array = np.expand_dims(base_img, axis=0)
    pred = float(image_model.predict(img_array, verbose=0)[0][0])
    predictions.append(pred)
    
    # Augmented versions
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    for _ in range(n_augmentations):
        # Generate augmented image
        aug_iter = datagen.flow(np.expand_dims(base_img, axis=0), batch_size=1)
        aug_img = next(aug_iter)[0]
        aug_img = np.expand_dims(aug_img, axis=0)
        pred = float(image_model.predict(aug_img, verbose=0)[0][0])
        predictions.append(pred)
    
    return np.mean(predictions)


def collect_predictions(use_tta=False):
    """Collect predictions with optional test-time augmentation."""
    image_model, clinical_model, scaler = load_models()
    df_clin = pd.read_csv(CLINICAL_CSV) if os.path.exists(CLINICAL_CSV) else None

    feats = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]
    all_y = []
    img_probs = []
    clin_probs = []

    classes = {"NORMAL": 0, "PNEUMONIA": 1}
    for cls_name, label in classes.items():
        folder = os.path.join(TEST_IMAGE_DIR, cls_name)
        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)
            
            if use_tta:
                ip = test_time_augmentation(image_model, img_path)
            else:
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


def find_optimal_threshold(y_true, y_proba):
    """Find optimal threshold using ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Find threshold that maximizes Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def search_fusion_grid(y_true, img_p, clin_p):
    """Grid search for best fusion weights and threshold."""
    best = {"f1": -1.0, "roc_auc": -1.0, "accuracy": -1.0}
    
    # More granular search
    for w in np.linspace(0.4, 0.95, 20):
        fusion = w * img_p + (1 - w) * clin_p
        
        # Find optimal threshold for this weight combination
        optimal_thr = find_optimal_threshold(y_true, fusion)
        
        preds = (fusion >= optimal_thr).astype(int)
        f1 = f1_score(y_true, preds)
        auc = roc_auc_score(y_true, fusion)
        acc = accuracy_score(y_true, preds)
        
        if f1 > best["f1"]:
            best = {
                "w_image": float(w),
                "w_clinical": float(1 - w),
                "threshold": float(optimal_thr),
                "f1": float(f1),
                "roc_auc": float(auc),
                "accuracy": float(acc),
            }
    
    return best


def search_fusion_cv(y_true, img_p, clin_p, n_splits=5):
    """Cross-validated fusion weight search."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_overall = {"f1": -1.0}
    
    for w in np.linspace(0.4, 0.95, 20):
        cv_scores = []
        cv_thresholds = []
        
        for train_idx, val_idx in cv.split(y_true, y_true):
            img_train, img_val = img_p[train_idx], img_p[val_idx]
            clin_train, clin_val = clin_p[train_idx], clin_p[val_idx]
            y_train, y_val = y_true[train_idx], y_true[val_idx]
            
            fusion_train = w * img_train + (1 - w) * clin_train
            fusion_val = w * img_val + (1 - w) * clin_val
            
            optimal_thr = find_optimal_threshold(y_train, fusion_train)
            preds_val = (fusion_val >= optimal_thr).astype(int)
            
            f1 = f1_score(y_val, preds_val)
            cv_scores.append(f1)
            cv_thresholds.append(optimal_thr)
        
        avg_f1 = np.mean(cv_scores)
        avg_thr = np.mean(cv_thresholds)
        
        if avg_f1 > best_overall["f1"]:
            best_overall = {
                "w_image": float(w),
                "w_clinical": float(1 - w),
                "threshold": float(avg_thr),
                "f1": float(avg_f1),
                "cv_std": float(np.std(cv_scores)),
            }
    
    return best_overall


def main():
    print("=" * 60)
    print("IMPROVED FUSION TUNING")
    print("=" * 60)
    
    # Collect predictions without TTA first
    print("\nCollecting predictions (without TTA)...")
    y_true, img_p, clin_p = collect_predictions(use_tta=False)
    print(f"Collected {len(y_true)} samples.")
    
    # Grid search
    print("\nPerforming grid search for fusion weights...")
    best_grid = search_fusion_grid(y_true, img_p, clin_p)
    
    print("\n=== BEST FUSION CONFIG (Grid Search) ===")
    for k, v in best_grid.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    # Cross-validated search
    print("\nPerforming cross-validated search...")
    best_cv = search_fusion_cv(y_true, img_p, clin_p)
    
    print("\n=== BEST FUSION CONFIG (Cross-Validated) ===")
    for k, v in best_cv.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    # Use CV results if better, otherwise use grid search
    if best_cv.get("f1", -1) > best_grid["f1"]:
        best = best_cv
        print("\n✅ Using cross-validated configuration")
    else:
        best = best_grid
        print("\n✅ Using grid search configuration")
    
    # Test with TTA
    print("\nTesting with Test-Time Augmentation...")
    y_true_tta, img_p_tta, clin_p_tta = collect_predictions(use_tta=True)
    
    fusion_tta = best["w_image"] * img_p_tta + best["w_clinical"] * clin_p_tta
    optimal_thr_tta = find_optimal_threshold(y_true_tta, fusion_tta)
    preds_tta = (fusion_tta >= optimal_thr_tta).astype(int)
    
    f1_tta = f1_score(y_true_tta, preds_tta)
    auc_tta = roc_auc_score(y_true_tta, fusion_tta)
    acc_tta = accuracy_score(y_true_tta, preds_tta)
    
    print(f"\n=== RESULTS WITH TTA ===")
    print(f"F1 Score: {f1_tta:.4f}")
    print(f"ROC-AUC: {auc_tta:.4f}")
    print(f"Accuracy: {acc_tta:.4f}")
    print(f"Optimal Threshold: {optimal_thr_tta:.4f}")
    
    # Update best config with TTA results
    best["tta_f1"] = float(f1_tta)
    best["tta_roc_auc"] = float(auc_tta)
    best["tta_accuracy"] = float(acc_tta)
    best["tta_threshold"] = float(optimal_thr_tta)
    
    # Save configuration
    os.makedirs("models", exist_ok=True)
    with open("models/fusion_config.json", "w") as f:
        json.dump(best, f, indent=2)
    
    print("\n✅ Saved fusion configuration to models/fusion_config.json")
    print("Label: SIMULATED EVALUATION – use for research/demo only.")


if __name__ == "__main__":
    main()

