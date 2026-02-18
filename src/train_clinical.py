import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb


print("STEP 1: Loading dataset...")
df = pd.read_csv("data/clinical/clinical_data.csv")

features = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]
X = df[features]
y = df["label"]

print("STEP 2: Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("STEP 3: Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = pred
    
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba)
    }


results = {}

print("\nTraining RandomForest...")
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_scaled, y_train)
results["RandomForest"] = evaluate(rf, X_test_scaled, y_test)

print("Training CatBoost...")
cb = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=6,
    verbose=False
)
cb.fit(X_train, y_train)  
results["CatBoost"] = evaluate(cb, X_test, y_test)

print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)
xgb.fit(X_train_scaled, y_train)
results["XGBoost"] = evaluate(xgb, X_test_scaled, y_test)

print("Training LightGBM...")
lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    objective="binary"
)
lgbm.fit(X_train_scaled, y_train)
results["LightGBM"] = evaluate(lgbm, X_test_scaled, y_test)

print("\n=== MODEL SCORES ===")
for model_name, metrics in results.items():
    print(f"\n📌 {model_name}:")
    print(f"   Accuracy = {metrics['accuracy']:.4f}")
    print(f"   F1 Score = {metrics['f1']:.4f}")
    print(f"   AUC Score = {metrics['auc']:.4f}")

best_model_name = max(results, key=lambda m: results[m]["accuracy"])
best_model = {
    "RandomForest": rf,
    "CatBoost": cb,
    "XGBoost": xgb,
    "LightGBM": lgbm,
}[best_model_name]

print(f"\n✅ BEST MODEL SELECTED: {best_model_name}")

pickle.dump(best_model, open("models/clinical_best.pkl", "wb"))
pickle.dump(scaler, open("models/clinical_best_scaler.pkl", "wb"))

print("\n🎉 Training Complete! Best model saved as clinical_best.pkl")
