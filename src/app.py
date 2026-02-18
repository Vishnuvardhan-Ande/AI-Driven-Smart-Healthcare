import os
import io
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import keras
import shap
import cv2
import traceback
import matplotlib
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask import flash
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, redirect, url_for, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from xml.sax.saxutils import escape
import re

# PATHS & FLASK SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)   # ai-healthcare
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clinical")
USERS_FILE = os.path.join(PROJECT_ROOT, "users.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'healthcare-secret-key-2024'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

# User class
class User(UserMixin):
    def __init__(self, id, email, name):
        self.id = id
        self.email = email
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
            for user in users:
                if user['id'] == user_id:
                    return User(user['id'], user['email'], user['name'])
    return None

# Initialize users file
def init_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)

init_users_file()

print("PROJECT_ROOT:", PROJECT_ROOT)
print("TEMPLATES_DIR:", TEMPLATES_DIR)
print("STATIC_DIR:", STATIC_DIR)

# LOAD MODELS
print("Loading image model...")

class PatchedInputLayer(keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_input_shape" not in config and "shape" not in config:
            config["batch_input_shape"] = (None, 224, 224, 3)
        return super().from_config(config)

image_model = keras.models.load_model(
    os.path.join(MODELS_DIR, "dense_best.h5"),
    custom_objects={"InputLayer": PatchedInputLayer},
    compile=False,
)

print("Loading clinical model & scaler...")
with open(os.path.join(MODELS_DIR, "clinical_best.pkl"), "rb") as f:
    clinical_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "clinical_best_scaler.pkl"), "rb") as f:
    clinical_scaler = pickle.load(f)

explainer = shap.TreeExplainer(clinical_model)

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
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer
            break
    if last_conv is None:
        raise RuntimeError("No Conv2D layer found in model for Grad-CAM.")

    grad_model = keras.models.Model(inputs=image_model.inputs,
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
def _to_paragraph_text(text: str) -> str:
    """Convert plain/markdown-ish text to safe Paragraph markup."""
    if not text:
        return ""
    # Basic markdown bold -> reportlab <b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Strip emoji bullets that can render as tofu in PDF fonts
    text = (
        text.replace("🔴 ", "")
        .replace("🟠 ", "")
        .replace("🟡 ", "")
        .replace("🟢 ", "")
    )
    # Escape XML entities, keep our <b> tags intact by escaping first then unescaping tags
    text = escape(text)
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    # Preserve newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
    return text


def _scaled_rl_image(path: str, max_width: float, max_height: float) -> RLImage:
    reader = ImageReader(path)
    iw, ih = reader.getSize()
    if not iw or not ih:
        return RLImage(path, width=max_width, height=max_height)
    scale = min(max_width / float(iw), max_height / float(ih))
    return RLImage(path, width=float(iw) * scale, height=float(ih) * scale)


def generate_pdf_report(
    patient_info,
    image_rel,
    gradcam_rel,
    shap_summary_rel,
    shap_wf_rel,
    predictions,
    explanation_text,
    final_diagnosis=None,
    fusion_pred=None,
    risk_band=None,
    out_name="report.pdf",
):
    out_path = os.path.join(OUTPUTS_DIR, out_name)
    # Slightly tighter margins helps keep sections + images together
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Healthcare Diagnosis Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Patient & Input Details:", styles["Heading2"]))
    for k,v in patient_info.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    story.append(Spacer(1,12))

    story.append(Paragraph("Predictions:", styles["Heading2"]))
    for k,v in predictions.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    story.append(Spacer(1,12))

    story.append(Paragraph("Model Explanation (short):", styles["Heading2"]))
    story.append(Paragraph(_to_paragraph_text(explanation_text), styles["Normal"]))
    story.append(Spacer(1,12))

    # Add images (scaled)
    image_paths = []
    if image_rel: image_paths.append((os.path.join(PROJECT_ROOT, image_rel.lstrip("/")), "Original X-ray"))
    if gradcam_rel: image_paths.append((os.path.join(PROJECT_ROOT, gradcam_rel.lstrip("/")), "Grad-CAM overlay"))
    if shap_summary_rel: image_paths.append((os.path.join(PROJECT_ROOT, shap_summary_rel.lstrip("/")), "SHAP summary"))
    if shap_wf_rel: image_paths.append((os.path.join(PROJECT_ROOT, shap_wf_rel.lstrip("/")), "SHAP waterfall"))

    # Keep images and headings together and scaled to avoid page splits
    max_w = 6.8 * inch
    max_h = 3.2 * inch

    for p, title in image_paths:
        try:
            block = [
                Paragraph(title, styles["Heading3"]),
                Spacer(1, 6),
                _scaled_rl_image(p, max_width=max_w, max_height=max_h),
                Spacer(1, 10),
            ]
            story.append(KeepTogether(block))
        except Exception as e:
            print("Skipping image in PDF due to:", e)

    if final_diagnosis:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Final Interpretation:", styles["Heading2"]))
        story.append(Paragraph(_to_paragraph_text(final_diagnosis), styles["Normal"]))

    # Always end with a concise final prediction block
    if fusion_pred is not None or risk_band:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Final Prediction:", styles["Heading2"]))
        lines = []
        if fusion_pred is not None:
            try:
                lines.append(f"<b>Fusion probability:</b> {float(fusion_pred) * 100:.1f}% ({float(fusion_pred):.3f})")
            except Exception:
                lines.append(f"<b>Fusion probability:</b> {fusion_pred}")
        if risk_band:
            lines.append(f"<b>Risk band:</b> {escape(str(risk_band))}")
        story.append(Paragraph("<br/>".join(lines), styles["Normal"]))

    doc.build(story)
    return "/static/outputs/" + out_name

# Short explanation generator
def make_explanation_text(pred_img, pred_clinical, fusion, shap_wf_rel):
    try:
        important = []
        text = (f"The image model predicts probability {pred_img:.3f} and the clinical model predicts {pred_clinical:.3f}. "
                f"The fused score (average) is {fusion:.3f}. The top contributing clinical features (from SHAP) "
                f"are shown in the SHAP plots. Positive SHAP bars increase the predicted risk while negative reduce it. "
                "Please see the SHAP waterfall and summary plots for per-feature detail and relative importance.")
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

# Authentication Routes
@app.route("/auth", methods=["GET"])
def auth():
    return render_template("auth.html")


@app.route("/signup", methods=["POST"])
def signup():
    # Support both old and new field names from the template
    name = (request.form.get("name") or request.form.get("signup_name") or "").strip()
    email = (request.form.get("email") or request.form.get("signup_email") or "").strip()
    password = (request.form.get("password") or request.form.get("signup_password") or "").strip()
    confirm_password = (
        request.form.get("confirm_password")
        or request.form.get("signup_confirm_password")
        or ""
    ).strip()

    if not name or not email or not password:
        flash("All fields are required for signup.", "danger")
        return redirect(url_for("auth"))

    if confirm_password and password != confirm_password:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("auth"))

    if len(password) < 6:
        flash("Password must be at least 6 characters.", "danger")
        return redirect(url_for("auth"))

    with open(USERS_FILE, "r") as f:
        users = json.load(f)

    for user in users:
        if user["email"].lower() == email.lower():
            flash("Email already registered. Please log in instead.", "danger")
            return redirect(url_for("auth"))

    user_id = str(len(users) + 1)
    new_user = {
        "id": user_id,
        "email": email,
        "name": name,
        "password_hash": generate_password_hash(password),
    }
    users.append(new_user)

    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

    user = User(user_id, email, name)
    login_user(user)
    flash("Account created. You are now logged in.", "success")
    return redirect(url_for("home"))


@app.route("/login", methods=["POST"])
def login():
    # Support both old and new field names from the template
    email = (request.form.get("email") or request.form.get("login_email") or "").strip()
    password = (request.form.get("password") or request.form.get("login_password") or "").strip()

    if not email or not password:
        flash("Email and password are required.", "danger")
        return redirect(url_for("auth"))

    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    except FileNotFoundError:
        users = []

    for user_data in users:
        if user_data["email"].lower() == email.lower():
            if check_password_hash(user_data["password_hash"], password):
                user = User(user_data["id"], user_data["email"], user_data["name"])
                login_user(user)
                flash("Logged in successfully.", "success")
                return redirect(url_for("home"))
            else:
                flash("Invalid password.", "danger")
                return redirect(url_for("auth"))

    flash("Email not found. Please sign up first.", "danger")
    return redirect(url_for("auth"))


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("auth"))


@app.route("/")
def home():
    if not current_user.is_authenticated:
        return redirect(url_for("auth"))
    user_initial = current_user.name[0].upper() if current_user.name else "U"
    return render_template(
        "index.html",
        user=current_user,
        user_name=current_user.name,
        user_initial=user_initial,
    )

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
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

        fusion_pred = (0.85 * img_pred) + (0.15 * clinical_pred)

        final_diagnosis = generate_final_diagnosis(img_pred, clinical_pred, fusion_pred)

        patient_name = (request.form.get("patient_name") or "").strip()

        # Derive a simple risk band label for the UI
        fusion_p = fusion_pred * 100.0
        if fusion_p >= 80:
            risk_band = "High risk"
        elif fusion_p >= 60:
            risk_band = "Moderate risk"
        elif fusion_p >= 30:
            risk_band = "Low–moderate risk"
        else:
            risk_band = "Very low risk"

        gradcam_rel = generate_gradcam_overlay(upload_path, out_name=f"gradcam_{upload_name}.png")

        shap_summary_rel, shap_wf_rel = generate_shap_plots(clinical_values, sample_name_prefix=f"shap_{upload_name}")

        explanation_text = make_explanation_text(img_pred, clinical_pred, fusion_pred, shap_wf_rel)

        patient_info = {
            "Patient name": patient_name if patient_name else "N/A",
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

        pdf_rel = generate_pdf_report(
            patient_info,
            image_rel="/static/uploads/" + upload_name,
            gradcam_rel=gradcam_rel,
            shap_summary_rel=shap_summary_rel,
            shap_wf_rel=shap_wf_rel,
            predictions=predictions,
            explanation_text=explanation_text,
            final_diagnosis=final_diagnosis,
            fusion_pred=fusion_pred,
            risk_band=risk_band,
            out_name=f"report_{upload_name}.pdf",
        )

        # Render the main dashboard with results instead of returning raw JSON
        user_initial = current_user.name[0].upper() if current_user.name else "U"
        return render_template(
            "index.html",
            user=current_user,
            user_name=current_user.name,
            user_initial=user_initial,
            img_path="/static/uploads/" + upload_name,
            gradcam_path=gradcam_rel,
            shap_summary_path=shap_summary_rel,
            shap_waterfall_path=shap_wf_rel,
            img_pred=img_pred,
            clinical_pred=clinical_pred,
            fusion_pred=fusion_pred,
            explanation_text=explanation_text,
            pdf_path=pdf_rel,
            final_diagnosis=final_diagnosis,
            risk_band=risk_band,
        )
    except Exception as e:
        print("REAL ERROR:", traceback.format_exc())
        flash(f"An error occurred while running prediction: {e}", "danger")
        return redirect(url_for("home"))

@app.route("/download_report")
@login_required
def download_report():
    files = [os.path.join(OUTPUTS_DIR, f) for f in os.listdir(OUTPUTS_DIR) if f.endswith(".pdf")]
    if not files:
        return "No report available", 404
    latest = max(files, key=os.path.getctime)
    return send_file(latest, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
