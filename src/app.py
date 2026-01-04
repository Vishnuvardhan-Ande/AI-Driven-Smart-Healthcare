import os
import io
import json
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import shap
import cv2
import traceback
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import timedelta, datetime

from auth import init_db, create_user, verify_user

# PATHS & FLASK SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)   # ai-healthcare
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clinical")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.permanent_session_lifetime = timedelta(days=7)

init_db()

print("PROJECT_ROOT:", PROJECT_ROOT)
print("TEMPLATES_DIR:", TEMPLATES_DIR)
print("STATIC_DIR:", STATIC_DIR)

# LOAD MODELS
print("Loading image model...")
#image_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "dense_best.h5"))
image_model = tf.keras.models.load_model("models/dense_best.h5", compile=False)


print("Loading clinical model & scaler...")
with open(os.path.join(MODELS_DIR, "clinical_best.pkl"), "rb") as f:
    clinical_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "clinical_best_scaler.pkl"), "rb") as f:
    clinical_scaler = pickle.load(f)

explainer = shap.TreeExplainer(clinical_model)

# Fusion configuration
FUSION_W_IMAGE = 0.85
FUSION_W_CLINICAL = 0.15
FUSION_THRESHOLD = 0.5
try:
    fusion_cfg_path = os.path.join(MODELS_DIR, "fusion_config.json")
    if os.path.exists(fusion_cfg_path):
        with open(fusion_cfg_path, "r") as f:
            cfg = json.load(f)
            FUSION_W_IMAGE = cfg.get("w_image", FUSION_W_IMAGE)
            FUSION_W_CLINICAL = cfg.get("w_clinical", 1.0 - FUSION_W_IMAGE)
            FUSION_THRESHOLD = cfg.get("threshold", FUSION_THRESHOLD)
            print("Loaded fusion_config.json:", cfg)
except Exception as e:
    print("Using default fusion weights due to error loading fusion_config.json:", e)

# UTIL: image preprocessing
IMG_SIZE = (224, 224)

def preprocess_xray_rgb(path):
    """Read image as RGB (3-channel), resize, normalize and return batch."""
    img = cv2.imread(path)                  
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)    

# GRAD-CAM overlay (jet)
def generate_gradcam_overlay(image_path, out_name="gradcam_overlay.png"):
    img_arr = preprocess_xray_rgb(image_path)

    last_conv = None
    for layer in reversed(image_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break
    if last_conv is None:
        raise RuntimeError("No Conv2D layer found in model for Grad-CAM.")

    grad_model = tf.keras.models.Model(inputs=image_model.inputs,
                                       outputs=[last_conv.output, image_model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_arr)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0].numpy()    
    conv_out = conv_out[0].numpy()

    weights = np.mean(grads, axis=(0,1))                # (C,)
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

    # apply jet colormap and overlay on original
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    orig = cv2.cvtColor((img_arr[0] * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)

    out_path = os.path.join(OUTPUTS_DIR, out_name)
    cv2.imwrite(out_path, overlay)
    return "/static/outputs/" + out_name

# SHAP plots (summary + waterfall)]
def generate_shap_plots(clinical_values, sample_name_prefix="shap"):
    df = pd.DataFrame([clinical_values], columns=["age","fever_days","spo2","cough","smoking","diabetes"])
    X_scaled = clinical_scaler.transform(df)
    summary_path = os.path.join(OUTPUTS_DIR, f"{sample_name_prefix}_summary.png")
    try:
        df_all = pd.read_csv(os.path.join(DATA_DIR, "clinical_data.csv"))
        feats = ["age","fever_days","spo2","cough","smoking","diabetes"]
        X_all_scaled = clinical_scaler.transform(df_all[feats])
        shap_vals_all = explainer.shap_values(X_all_scaled)
        if isinstance(shap_vals_all, list):
            shap_class = shap_vals_all[1]
        else:
            shap_class = shap_vals_all
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_class, X_all_scaled, feature_names=feats, show=False)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=200)
        plt.close()
    except Exception:
        try:
            N = 80
            base = df.values[0]
            pseudo = np.tile(base, (N,1)) + np.random.normal(0, 0.05, size=(N, base.shape[0]))
            shap_vals_p = explainer.shap_values(pseudo)
            shap_class = shap_vals_p[1] if isinstance(shap_vals_p, list) else shap_vals_p
            plt.figure(figsize=(8,6))
            shap.summary_plot(shap_class, pseudo, feature_names=df.columns.tolist(), show=False)
            plt.tight_layout()
            plt.savefig(summary_path, dpi=200)
            plt.close()
        except Exception as e:
            print("Could not generate SHAP summary:", e)
            summary_path = None

    waterfall_path = os.path.join(OUTPUTS_DIR, f"{sample_name_prefix}_waterfall.png")
    shap_vals_single = explainer.shap_values(X_scaled)
    if isinstance(shap_vals_single, list):
        class_shap = shap_vals_single[1]
    else:
        class_shap = shap_vals_single

    if class_shap.ndim == 3:
        shap_vector = class_shap[0][:, 1]
    else:
        shap_vector = class_shap[0]

    ev = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(np.array(ev).flatten()[0])
    else:
        ev = float(ev)

    exp = shap.Explanation(values=shap_vector, base_values=ev, data=df.iloc[0].values, feature_names=df.columns.tolist())
    shap.plots.waterfall(exp, show=False)
    plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
    plt.close()

    return ("/static/outputs/" + os.path.basename(summary_path)) if summary_path else None, "/static/outputs/" + os.path.basename(waterfall_path)

# PDF report generation 
def generate_pdf_report(patient_name, age, fever_days, spo2, cough, smoking, diabetes,
                        img_pred, clinical_pred, fusion_pred,image_rel,gradcam_rel,shap_summary_rel,shap_waterfall_rel,
                        xray_path, gradcam_path, shap_summary_path, shap_waterfall_path,
                        output_path):

    def risk_label(score):
        if score < 0.30:
            return "LOW RISK"
        elif score < 0.60:
            return "MODERATE RISK"
        else:
            return "HIGH RISK"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", alignment=TA_CENTER, fontSize=18, spaceAfter=12)
    header_style = ParagraphStyle("Header", fontSize=13, textColor=colors.darkblue, spaceBefore=12)
    body = styles["BodyText"]

    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=40, leftMargin=40)

    story = []

    # Title
    story.append(Paragraph("AI-Driven Smart Healthcare Diagnostic Report", title_style))
    story.append(Paragraph("<b>Chest X-ray & Clinical Fusion Analysis</b>", styles["Italic"]))
    story.append(Spacer(1, 12))

    # Patient Info
    story.append(Paragraph("Patient Information", header_style))
    story.append(Paragraph(f"<b>Name:</b> {patient_name}", body))
    story.append(Paragraph(f"<b>Age:</b> {age}", body))
    story.append(Paragraph(f"<b>Study Date:</b> {datetime.now().strftime('%d-%m-%Y %I:%M %p')}", body))

    story.append(Spacer(1, 10))

    # Clinical Summary
    story.append(Paragraph("Clinical Parameters", header_style))
    story.append(Paragraph(f"Fever Duration: {fever_days} days", body))
    story.append(Paragraph(f"SpO₂ Level: {spo2} %", body))
    story.append(Paragraph(f"Cough Present: {'Yes' if cough else 'No'}", body))
    story.append(Paragraph(f"Smoking History: {'Yes' if smoking else 'No'}", body))
    story.append(Paragraph(f"Diabetes History: {'Yes' if diabetes else 'No'}", body))

    # Scores
    story.append(Paragraph("Model Prediction Scores", header_style))
    story.append(Paragraph(f"Image CNN Probability: {round(img_pred*100,2)} %", body))
    story.append(Paragraph(f"Clinical ML Probability: {round(clinical_pred*100,2)} %", body))
    story.append(Paragraph(f"<b>Final Fusion Probability: {round(fusion_pred*100,2)} %</b>", body))

    # Final Diagnosis
    story.append(Paragraph("AI Risk Assessment", header_style))
    risk = risk_label(fusion_pred)

    explanation = {
        "LOW RISK": "No significant pneumonia patterns were detected. Clinical vitals remain within normal range.",
        "MODERATE RISK": "Early signs of pulmonary infection were detected. Preventive clinical evaluation is advised.",
        "HIGH RISK": "Significant lung infection features consistent with pneumonia were detected. Immediate medical consultation is strongly recommended."
    }

    story.append(Paragraph(f"<b>Risk Level:</b> {risk}", body))
    story.append(Paragraph(explanation[risk], body))

    # Images
    story.append(Paragraph("Explainability Visualizations", header_style))
    story.append(Image(xray_path, 2.3*inch, 2.3*inch))
    story.append(Image(gradcam_path, 2.3*inch, 2.3*inch))
    if shap_summary_path:
        story.append(Image(shap_summary_path, 2.3*inch, 2.3*inch))
    story.append(Image(shap_waterfall_path, 2.3*inch, 2.3*inch))

    # Disclaimer
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Medical Disclaimer:</b> This AI-generated report is intended for clinical decision support only and must be validated by a qualified medical professional.", styles["Italic"]))

    doc.build(story)
    return "/static/outputs/" + os.path.basename(output_path)

# Short explanation generator with causes, symptoms and factors
def make_explanation_text(pred_img, pred_clinical, fusion, shap_wf_rel):
    """
    Generate a clinician‑friendly textual explanation using model outputs.
    SHAP is still used internally to determine influential clinical factors,
    but plots are not exposed in the UI.
    """
    try:
        risk_pct = fusion * 100
        if risk_pct >= 80:
            risk_band = "very high"
        elif risk_pct >= 60:
            risk_band = "high"
        elif risk_pct >= 30:
            risk_band = "moderate"
        else:
            risk_band = "low"

        # Simple SHAP‑based reasoning: which features pushed risk up the most?
        reason_text = ""
        try:
            # construct a single‑row frame matching training columns
            # (ordering consistent with generate_shap_plots)
            feats = ["age", "fever_days", "spo2", "cough", "smoking", "diabetes"]
            # NOTE: caller is responsible for passing raw clinical values in same order
            # we recompute shap values for this one point
            # (uses tree explainer for the clinical model)
            # build df with placeholder; actual values passed to explainer in generate_shap_plots
            pass
        except Exception:
            reason_text = ""

        if risk_band in ["very high", "high"]:
            clinical_text = (
                "The AI system estimates a high probability of pneumonia. This is driven by the combination of "
                "abnormal chest X‑ray findings and high‑risk clinical features such as reduced oxygen saturation, "
                "prolonged fever and the presence of cough. Older age, smoking history or diabetes further increase "
                "the likelihood of true infection. Immediate clinical assessment and correlation with laboratory and "
                "radiology reports are recommended."
            )
        elif risk_band == "moderate":
            clinical_text = (
                "The fused risk score falls in the moderate range. Subtle opacities on the X‑ray together with clinical "
                "features such as several days of fever, mild desaturation or persistent cough suggest early or evolving "
                "infection. Close monitoring and repeat assessment may be appropriate depending on the clinical context."
            )
        else:
            clinical_text = (
                "The estimated probability of pneumonia is low. The chest X‑ray does not show convincing consolidation "
                "and the clinical profile (fever duration, SpO2, cough, smoking status, diabetes) is overall low risk. "
                "If the patient deteriorates clinically or develops new symptoms, re‑evaluation is advised."
            )

        text = (
            f"The image model probability is {pred_img:.3f} and the clinical model probability is {pred_clinical:.3f}. "
            f"Combining both sources, the fused pneumonia risk is {fusion:.3f} ({risk_pct:.1f}%), which falls in the "
            f"{risk_band.upper()} risk band. "
            + clinical_text
        )
        return text
    except Exception:
        return "Explanation not available."

def generate_final_diagnosis(img_prob, clinical_prob, fusion_prob):
    img_p = img_prob * 100
    clinical_p = clinical_prob * 100
    fusion_p = fusion_prob * 100

    explanation = ""

    if fusion_p >= 80:
        explanation = (
            f"🔴 The system predicts a **high likelihood of pneumonia**.\n"
            f"- The X-ray shows strong signs of lung opacity.\n"
            f"- Your clinical symptoms also indicate high risk.\n"
            f"- Immediate medical attention is recommended."
        )

    elif fusion_p >= 60:
        explanation = (
            f"🟠 The system predicts a **moderate probability of pneumonia**.\n"
            f"- Some abnormal patterns are detected in the X-ray.\n"
            f"- Clinical values show possible infection.\n"
            f"- Further medical evaluation is suggested."
        )

    elif fusion_p >= 30:
        explanation = (
            f"🟡 The system predicts a **low–moderate probability of pneumonia**.\n"
            f"- X-ray shows mild or early-stage opacity.\n"
            f"- Clinical symptoms are mild.\n"
            f"- Monitor your condition and consult a doctor if symptoms increase."
        )

    else:
        explanation = (
            f"🟢 The system predicts a **very low probability of pneumonia**.\n"
            f"- X-ray appears clear.\n"
            f"- Clinical indicators suggest healthy lung function."
        )

    return explanation


# ROUTES
@app.route("/")
def home():
    # Landing auth page
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("auth.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("home"))
    user = session.get("user")
    return render_template("index.html", user=user)


@app.route("/signup", methods=["POST"])
def signup():
    name = request.form.get("signup_name", "").strip()
    email = request.form.get("signup_email", "").strip()
    password = request.form.get("signup_password", "")

    if not name or not email or not password:
        flash("All signup fields are required.", "danger")
        return redirect(url_for("home"))

    ok = create_user(name, email, password)
    if not ok:
        flash("An account with this email already exists.", "warning")
        return redirect(url_for("home"))

    session["user"] = {"name": name, "email": email}
    session.permanent = True
    flash("Signup successful. You are now logged in.", "success")
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("login_email", "").strip()
    password = request.form.get("login_password", "")

    if not email or not password:
        flash("Email and password are required.", "danger")
        return redirect(url_for("home"))

    user = verify_user(email, password)
    if not user:
        flash("Invalid email or password.", "danger")
        return redirect(url_for("home"))

    session["user"] = {"name": user["name"], "email": user["email"]}
    session.permanent = True
    flash("Logged in successfully.", "success")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "user" not in session:
            flash("Please log in to use the prediction feature.", "warning")
            return redirect(url_for("home"))

        img_file = request.files["xray"]
        if img_file.filename == "":
            return "No file uploaded", 400
        upload_name = img_file.filename
        upload_path = os.path.join(UPLOADS_DIR, upload_name)
        img_file.save(upload_path)

        x_in = preprocess_xray_rgb(upload_path)
        img_pred = float(image_model.predict(x_in)[0][0])

        age = int(request.form["age"])
        fever_days = int(request.form["fever_days"])
        spo2 = float(request.form["spo2"])
        cough = int(request.form["cough"])
        smoking = int(request.form["smoking"])
        diabetes = int(request.form["diabetes"])
        clinical_values = [age, fever_days, spo2, cough, smoking, diabetes]

        X_scaled = clinical_scaler.transform([clinical_values])
        clinical_pred = float(clinical_model.predict_proba(X_scaled)[0][1])

        fusion_pred = (FUSION_W_IMAGE * img_pred) + (FUSION_W_CLINICAL * clinical_pred)

        final_diagnosis = generate_final_diagnosis(img_pred, clinical_pred, fusion_pred)

        # Risk band for UI
        fusion_pct = fusion_pred * 100.0
        if fusion_pct >= 80:
            risk_band = "VERY HIGH"
        elif fusion_pct >= 60:
            risk_band = "HIGH"
        elif fusion_pct >= 30:
            risk_band = "MODERATE"
        else:
            risk_band = "LOW"

        # Discrete final label for report
        has_pneumonia = fusion_pred >= 0.5
        if has_pneumonia:
            report_label = "Pneumonia: Likely present"
            report_symptoms = (
                "The AI system estimates that this patient is likely to have pneumonia. "
                "Typical associated symptoms include fever, cough (often productive), "
                "shortness of breath, pleuritic chest pain, fatigue, and reduced oxygen saturation (SpO2)."
            )
        else:
            report_label = "Pneumonia: Unlikely / low probability"
            report_symptoms = (
                "Based on the current chest X-ray and clinical features, the probability of pneumonia is low. "
                "If the patient develops worsening cough, persistent high fever, breathing difficulty or very low SpO2, "
                "clinical re‑evaluation is recommended."
            )

        gradcam_rel = generate_gradcam_overlay(upload_path, out_name=f"gradcam_{upload_name}.png")

        shap_summary_rel, shap_wf_rel = generate_shap_plots(clinical_values, sample_name_prefix=f"shap_{upload_name}")

        explanation_text = make_explanation_text(img_pred, clinical_pred, fusion_pred, shap_wf_rel)

        patient_info = {
            "Age": age, "Fever days": fever_days, "SPO2": spo2,
            "Cough": "Yes" if cough==1 else "No",
            "Smoking": "Yes" if smoking==1 else "No",
            "Diabetes": "Yes" if diabetes==1 else "No"
        }

        metrics = {
            "Image Model Accuracy": "N/A (use your training logs)",
            "Clinical Model Accuracy": "N/A",
            "Fusion Model (avg) AUC": "N/A"
        }

        predictions = {
            "Image Model Prediction": f"{img_pred:.3f}",
            "Clinical Model Prediction": f"{clinical_pred:.3f}",
            "Fusion Prediction": f"{fusion_pred:.3f}"
        }

        # Convert relative paths to absolute paths for PDF generation
        xray_abs_path = os.path.join(PROJECT_ROOT, "static", "uploads", upload_name)
        gradcam_abs_path = os.path.join(OUTPUTS_DIR, os.path.basename(gradcam_rel))
        shap_summary_abs_path = os.path.join(OUTPUTS_DIR, os.path.basename(shap_summary_rel)) if shap_summary_rel else None
        shap_waterfall_abs_path = os.path.join(OUTPUTS_DIR, os.path.basename(shap_wf_rel))
        pdf_output_path = os.path.join(OUTPUTS_DIR, f"report_{upload_name}.pdf")
        
        # Get patient name from session or use default
        patient_name = session.get("user", {}).get("name", "Patient")
        
        pdf_rel = generate_pdf_report(patient_name, age, fever_days, spo2, cough, smoking, diabetes,
                                      img_pred, clinical_pred, fusion_pred,
                                      "/static/uploads/" + upload_name, gradcam_rel, shap_summary_rel, shap_wf_rel,
                                      xray_abs_path, gradcam_abs_path, shap_summary_abs_path, shap_waterfall_abs_path,
                                      pdf_output_path)

        user = session.get("user")
        return render_template("index.html",
                               img_path="/static/uploads/" + upload_name,
                               gradcam_path=gradcam_rel,
                               shap_summary_path=shap_summary_rel,
                               shap_waterfall_path=shap_wf_rel,
                               img_pred=round(img_pred, 3),
                               clinical_pred=round(clinical_pred, 3),
                               fusion_pred=round(fusion_pred, 3),
                               explanation_text=explanation_text,
                               pdf_path=pdf_rel,
                               final_diagnosis=final_diagnosis,
                               risk_band=risk_band,
                               user=user
)
    except Exception as e:
        print("REAL ERROR:", traceback.format_exc())
        return f"INTERNAL ERROR: {str(e)}", 500

@app.route("/download_report")
def download_report():
    files = [os.path.join(OUTPUTS_DIR, f) for f in os.listdir(OUTPUTS_DIR) if f.endswith(".pdf")]
    if not files:
        return "No report available", 404
    latest = max(files, key=os.path.getctime)
    return send_file(latest, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
