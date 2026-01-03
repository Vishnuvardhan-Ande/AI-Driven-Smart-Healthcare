import os
import pandas as pd
import shap
import pickle
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def run_shap():

    print("STEP 0: Creating output folder if missing...")
    os.makedirs("outputs", exist_ok=True)

    print("STEP 1: Loading RandomForest clinical model...")
    with open("models/clinical_rf.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/clinical_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("STEP 2: Loading dataset...")
    df = pd.read_csv("data/clinical/clinical_data.csv")

    print("STEP 3: Selecting ALL features...")
    features = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]
    X = df[features]

    print("STEP 4: Scaling data for prediction...")
    X_scaled = scaler.transform(X)

    print("STEP 5: Creating SHAP explainer (force original behavior)...")
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")

    print("STEP 6: Computing SHAP values...")
    shap_values = explainer.shap_values(X_scaled, check_additivity=False)

    print("\n=== SHAP SHAPES CHECK ===")
    print("Raw SHAP shape:", np.array(shap_values).shape)
    print("X_scaled shape:", X_scaled.shape)
    print("=========================\n")

    shap_class = shap_values[:, :, 1]       # <-- THE FIX
    print("Using shap_class shape:", shap_class.shape)

    #  SHAP SUMMARY PLOT
    print("STEP 7: Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_class,
        X,
        feature_names=features,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=300)
    plt.close()

    # SHAP BAR PLOT
    print("STEP 8: Generating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_class,
        X,
        feature_names=features,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_bar.png", dpi=300)
    plt.close()

    # SHAP WATERFALL PLOT 
    print("STEP 9: Generating SHAP waterfall plot...")
    idx = 0  

    exp = shap.Explanation(
        values=shap_class[idx],
        base_values=explainer.expected_value[1],
        data=X.iloc[idx],
        feature_names=features
    )

    shap.plots.waterfall(exp, show=False)
    plt.savefig("outputs/shap_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ All SHAP plots generated successfully!")


if __name__ == "__main__":
    run_shap()
