"""
Clinical model training pipeline for pneumonia risk prediction.

This script:
  - Loads clinical features from data/clinical/clinical_data.csv
  - Performs feature engineering (risk bands and severity buckets)
  - Handles class imbalance with class weights
  - Trains multiple models with cross‑validation and basic hyperparameter search
  - Calibrates probabilities
  - Selects the best model by validation F1‑score
  - Saves the best model and scaler as clinical_best.pkl and clinical_best_scaler.pkl
  - Logs metrics to models/clinical_metrics.json
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

print("STEP 1: Loading dataset...")
df = pd.read_csv("data/clinical/clinical_data.csv")
df["label"] = df["label"].astype(int)

base_features = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]


def add_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Add clinically meaningful derived features."""
    frame = frame.copy()

    # SpO2 risk bands
    frame["spo2_risk_band"] = pd.cut(
        frame["spo2"],
        bins=[0, 88, 94, 100],
        labels=["severe_hypoxia", "mild_hypoxia", "normal"],
        include_lowest=True,
    )

    # Fever severity buckets
    frame["fever_bucket"] = pd.cut(
        frame["fever_days"],
        bins=[-1, 2, 7, 30],
        labels=["acute", "subacute", "prolonged"],
    )

    # Binary encodings
    frame["cough_bin"] = frame["cough"].astype(int)
    frame["smoking_bin"] = frame["smoking"].astype(int)
    frame["diabetes_bin"] = frame["diabetes"].astype(int)

    # One‑hot encode categorical engineered features
    frame = pd.get_dummies(
        frame,
        columns=["spo2_risk_band", "fever_bucket"],
        drop_first=True,
    )
    return frame


print("STEP 2: Feature engineering...")
df_fe = add_feature_engineering(df)
feature_cols = [c for c in df_fe.columns if c != "label"]
X = df_fe[feature_cols]
y = df_fe["label"]

print("STEP 3: Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("STEP 4: Scaling numerical features...")
num_cols = ["age", "fever_days", "spo2"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


def evaluate_model(model, X_eval, y_eval, name: str):
    """Return dictionary of standard classification metrics."""
    y_pred = model.predict(X_eval)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred)),
        "recall": float(recall_score(y_eval, y_pred)),
        "f1": float(f1_score(y_eval, y_pred)),
        "roc_auc": float(roc_auc_score(y_eval, y_proba)),
    }
    print(f"\n=== {name} (hold‑out metrics) ===")
    for k, v in metrics.items():
        print(f"{k:>9}: {v:.4f}")
    return metrics


def train_logistic(X_tr, y_tr):
    base = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    param_grid = {
        "C": np.logspace(-2, 2, 10),
        "penalty": ["l1", "l2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=12,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(best, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)
    return calibrated, search.best_params_


def train_random_forest(X_tr, y_tr):
    base = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=15,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=3)
    calibrated.fit(X_tr, y_tr)
    return calibrated, search.best_params_


def train_catboost(X_tr, y_tr):
    # CatBoost handles imbalance with scale_pos_weight
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        scale_pos_weight=pos_weight,
        verbose=False,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X_tr, y_tr):
        X_tr_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        model.fit(X_tr_fold, y_tr_fold, eval_set=(X_val_fold, y_val_fold), verbose=False)
        y_val_pred = model.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, y_val_pred))
    print(f"CatBoost 5‑fold F1 (mean±std): {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    model.fit(X_tr, y_tr, verbose=False)
    return model, {"note": "CatBoost with fixed hyperparameters, CV‑validated"}


models = {}
metrics_log = {}
params_log = {}

print("\nSTEP 5: Training Logistic Regression (with calibration)...")
log_reg_model, log_reg_params = train_logistic(X_train, y_train)
models["LogisticRegression"] = log_reg_model
params_log["LogisticRegression"] = log_reg_params
metrics_log["LogisticRegression"] = evaluate_model(log_reg_model, X_test, y_test, "LogisticRegression")

print("\nSTEP 6: Training Random Forest (with calibration)...")
rf_model, rf_params = train_random_forest(X_train, y_train)
models["RandomForest"] = rf_model
params_log["RandomForest"] = rf_params
metrics_log["RandomForest"] = evaluate_model(rf_model, X_test, y_test, "RandomForest")

print("\nSTEP 7: Training CatBoost...")
cb_model, cb_params = train_catboost(X_train, y_train)
models["CatBoost"] = cb_model
params_log["CatBoost"] = cb_params
metrics_log["CatBoost"] = evaluate_model(cb_model, X_test, y_test, "CatBoost")

print("\n=== MODEL COMPARISON (by F1 on hold‑out) ===")
for name, m in metrics_log.items():
    print(f"{name:>18}: F1 = {m['f1']:.4f}, AUC = {m['roc_auc']:.4f}")

best_name = max(metrics_log, key=lambda n: metrics_log[n]["f1"])
best_model = models[best_name]

print(f"\n✅ BEST MODEL SELECTED: {best_name}")

with open("models/clinical_best.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/clinical_best_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

metrics_payload = {
    "metrics": metrics_log,
    "hyperparameters": params_log,
    "best_model": best_name,
}
with open("models/clinical_metrics.json", "w") as f:
    json.dump(metrics_payload, f, indent=2)

print("\n🎉 Training complete. Best model and scaler saved as clinical_best.pkl / clinical_best_scaler.pkl")
print("📄 Detailed metrics written to models/clinical_metrics.json")
