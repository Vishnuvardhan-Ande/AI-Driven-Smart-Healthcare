"""
Improved clinical model training with better hyperparameter tuning and additional models.

Improvements:
- XGBoost and LightGBM models added
- More extensive hyperparameter search
- Better feature engineering
- Ensemble methods
- Improved cross-validation strategy
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
import os

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

print("STEP 1: Loading dataset...")
df = pd.read_csv("data/clinical/clinical_data.csv")
df["label"] = df["label"].astype(int)

base_features = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]


def add_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with more clinically meaningful features."""
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

    # Age groups (clinically relevant)
    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[0, 18, 40, 60, 100],
        labels=["pediatric", "young_adult", "middle_age", "elderly"],
        include_lowest=True,
    )

    # Interaction features
    frame["age_fever_interaction"] = frame["age"] * frame["fever_days"]
    frame["spo2_age_interaction"] = frame["spo2"] * frame["age"]
    frame["risk_score"] = (
        (100 - frame["spo2"]) * 0.5 +
        frame["fever_days"] * 0.3 +
        frame["age"] * 0.01 +
        frame["cough"].astype(int) * 10 +
        frame["smoking"].astype(int) * 15 +
        frame["diabetes"].astype(int) * 20
    )

    # Binary encodings
    frame["cough_bin"] = frame["cough"].astype(int)
    frame["smoking_bin"] = frame["smoking"].astype(int)
    frame["diabetes_bin"] = frame["diabetes"].astype(int)

    # One-hot encode categorical engineered features
    frame = pd.get_dummies(
        frame,
        columns=["spo2_risk_band", "fever_bucket", "age_group"],
        drop_first=True,
    )
    return frame


print("STEP 2: Enhanced feature engineering...")
df_fe = add_feature_engineering(df)
feature_cols = [c for c in df_fe.columns if c != "label"]
X = df_fe[feature_cols]
y = df_fe["label"]

print(f"Total features after engineering: {len(feature_cols)}")

print("STEP 3: Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("STEP 4: Scaling numerical features...")
num_cols = ["age", "fever_days", "spo2", "age_fever_interaction", 
            "spo2_age_interaction", "risk_score"]
# Only scale columns that exist
num_cols = [col for col in num_cols if col in X_train.columns]
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
    print(f"\n=== {name} (hold-out metrics) ===")
    for k, v in metrics.items():
        print(f"{k:>9}: {v:.4f}")
    return metrics


def train_logistic(X_tr, y_tr):
    """Improved Logistic Regression with extensive hyperparameter search."""
    base = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
    )
    param_grid = {
        "C": np.logspace(-3, 2, 20),
        "penalty": ["l1", "l2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=30,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(best, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)
    return calibrated, search.best_params_


def train_random_forest(X_tr, y_tr):
    """Improved Random Forest with better hyperparameter search."""
    base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [300, 400, 500, 600],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=40,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=5)
    calibrated.fit(X_tr, y_tr)
    return calibrated, search.best_params_


def train_catboost(X_tr, y_tr):
    """Improved CatBoost with hyperparameter tuning."""
    if not CATBOOST_AVAILABLE:
        return None, None
    
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    
    param_grid = {
        "iterations": [400, 500, 600],
        "learning_rate": [0.03, 0.05, 0.07],
        "depth": [5, 6, 7],
        "l2_leaf_reg": [1, 3, 5],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_score = -1
    best_params = None
    best_model = None
    
    # Manual grid search for CatBoost (better than RandomizedSearchCV)
    for iterations in param_grid["iterations"]:
        for lr in param_grid["learning_rate"]:
            for depth in param_grid["depth"]:
                for l2 in param_grid["l2_leaf_reg"]:
                    scores = []
                    for train_idx, val_idx in cv.split(X_tr, y_tr):
                        X_tr_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
                        y_tr_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
                        
                        model = CatBoostClassifier(
                            iterations=iterations,
                            learning_rate=lr,
                            depth=depth,
                            l2_leaf_reg=l2,
                            loss_function="Logloss",
                            eval_metric="AUC",
                            random_seed=RANDOM_STATE,
                            scale_pos_weight=pos_weight,
                            verbose=False,
                        )
                        model.fit(X_tr_fold, y_tr_fold, eval_set=(X_val_fold, y_val_fold), verbose=False)
                        y_val_pred = model.predict(X_val_fold)
                        scores.append(f1_score(y_val_fold, y_val_pred))
                    
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            "iterations": iterations,
                            "learning_rate": lr,
                            "depth": depth,
                            "l2_leaf_reg": l2,
                        }
    
    print(f"Best CatBoost CV F1: {best_score:.4f} with params: {best_params}")
    
    model = CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        scale_pos_weight=pos_weight,
        verbose=False,
    )
    model.fit(X_tr, y_tr, verbose=False)
    return model, best_params


def train_xgboost(X_tr, y_tr):
    """XGBoost with hyperparameter tuning."""
    if not XGBOOST_AVAILABLE:
        return None, None
    
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    
    base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
    )
    
    param_grid = {
        "n_estimators": [300, 400, 500],
        "max_depth": [4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.07],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=50,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(best, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)
    return calibrated, search.best_params_


def train_lightgbm(X_tr, y_tr):
    """LightGBM with hyperparameter tuning."""
    if not LIGHTGBM_AVAILABLE:
        return None, None
    
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    
    base = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    
    param_grid = {
        "n_estimators": [300, 400, 500],
        "max_depth": [4, 5, 6, 7, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.07],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=50,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(best, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)
    return calibrated, search.best_params_


# Train all models
models = {}
metrics_log = {}
params_log = {}

print("\nSTEP 5: Training Logistic Regression...")
log_reg_model, log_reg_params = train_logistic(X_train, y_train)
if log_reg_model:
    models["LogisticRegression"] = log_reg_model
    params_log["LogisticRegression"] = log_reg_params
    metrics_log["LogisticRegression"] = evaluate_model(log_reg_model, X_test, y_test, "LogisticRegression")

print("\nSTEP 6: Training Random Forest...")
rf_model, rf_params = train_random_forest(X_train, y_train)
if rf_model:
    models["RandomForest"] = rf_model
    params_log["RandomForest"] = rf_params
    metrics_log["RandomForest"] = evaluate_model(rf_model, X_test, y_test, "RandomForest")

if CATBOOST_AVAILABLE:
    print("\nSTEP 7: Training CatBoost...")
    cb_model, cb_params = train_catboost(X_train, y_train)
    if cb_model:
        models["CatBoost"] = cb_model
        params_log["CatBoost"] = cb_params
        metrics_log["CatBoost"] = evaluate_model(cb_model, X_test, y_test, "CatBoost")

if XGBOOST_AVAILABLE:
    print("\nSTEP 8: Training XGBoost...")
    xgb_model, xgb_params = train_xgboost(X_train, y_train)
    if xgb_model:
        models["XGBoost"] = xgb_model
        params_log["XGBoost"] = xgb_params
        metrics_log["XGBoost"] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

if LIGHTGBM_AVAILABLE:
    print("\nSTEP 9: Training LightGBM...")
    lgb_model, lgb_params = train_lightgbm(X_train, y_train)
    if lgb_model:
        models["LightGBM"] = lgb_model
        params_log["LightGBM"] = lgb_params
        metrics_log["LightGBM"] = evaluate_model(lgb_model, X_test, y_test, "LightGBM")

print("\n=== MODEL COMPARISON (by F1 on hold-out) ===")
for name, m in metrics_log.items():
    print(f"{name:>18}: F1 = {m['f1']:.4f}, AUC = {m['roc_auc']:.4f}, Acc = {m['accuracy']:.4f}")

# Select best model
best_name = max(metrics_log, key=lambda n: metrics_log[n]["f1"])
best_model = models[best_name]

print(f"\n✅ BEST MODEL SELECTED: {best_name}")
print(f"   F1 Score: {metrics_log[best_name]['f1']:.4f}")
print(f"   ROC-AUC: {metrics_log[best_name]['roc_auc']:.4f}")
print(f"   Accuracy: {metrics_log[best_name]['accuracy']:.4f}")

# Optional: Create ensemble of top 3 models
if len(models) >= 3:
    print("\nCreating ensemble of top 3 models...")
    sorted_models = sorted(metrics_log.items(), key=lambda x: x[1]['f1'], reverse=True)
    top_3_names = [name for name, _ in sorted_models[:3]]
    
    ensemble_models = [(name, models[name]) for name in top_3_names]
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='soft',
        weights=[metrics_log[name]['f1'] for name in top_3_names]
    )
    ensemble.fit(X_train, y_train)
    
    ensemble_metrics = evaluate_model(ensemble, X_test, y_test, "Ensemble (Top 3)")
    
    # Use ensemble if it's better
    if ensemble_metrics['f1'] > metrics_log[best_name]['f1']:
        print(f"\n✅ Ensemble performs better! Using ensemble as best model.")
        best_model = ensemble
        best_name = "Ensemble"
        metrics_log["Ensemble"] = ensemble_metrics

# Save best model
os.makedirs("models", exist_ok=True)
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

