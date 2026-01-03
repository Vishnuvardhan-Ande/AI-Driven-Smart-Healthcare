import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("STEP 1: Loading dataset...")
df = pd.read_csv("data/clinical/clinical_data.csv")

print("STEP 2: Fixing label type...")
df["label"] = df["label"].astype(int)   

print("STEP 3: Selecting features and label...")
features = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]
X = df[features]
y = df["label"]

print("STEP 4: Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("STEP 5: Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("STEP 6: Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("STEP 7: Saving model and scaler...")
pickle.dump(model, open("models/clinical_rf.pkl", "wb"))
pickle.dump(scaler, open("models/clinical_scaler.pkl", "wb"))

print("✅ Model re-trained correctly with BINARY labels!")
