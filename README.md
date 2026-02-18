# AI-Driven Smart Healthcare Diagnosis Framework

## 📌 Overview
An AI-powered healthcare system that predicts pneumonia using:
- Chest X-ray images (CNN)
- Clinical features (ML models)
- Explainability via SHAP & Grad-CAM

## 🚀 Features
- Image-based pneumonia prediction
- Clinical risk analysis
- Fusion-based diagnosis
- Explainable AI (XAI)
- PDF medical report generation
- Flask-based web UI

## 🛠 Tech Stack
- Python, Flask
- TensorFlow, Keras
- XGBoost, CatBoost, LightGBM
- SHAP, Grad-CAM
- HTML, CSS, Bootstrap

## 📂 Project Structure

ai-healthcare/
├── src/
├── models/
├── data/
├── templates/
├── static/
└── README.md

## ▶️ How to Run

### Quick Start
**Windows:**
```bash
start_app.bat
```

**Linux/Mac:**
```bash
python src/app.py
```

### Troubleshooting

**If you see "Image model is not loaded" error:**

1. **Check environment variables:** Make sure `SKIP_IMAGE_MODEL` is not set to `1`
   ```bash
   # Windows PowerShell
   $env:SKIP_IMAGE_MODEL = ""
   
   # Windows CMD
   set SKIP_IMAGE_MODEL=
   
   # Linux/Mac
   unset SKIP_IMAGE_MODEL
   ```

2. **Verify TensorFlow installation:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```
   Should show `2.16.1` or higher.

3. **Check model file exists:**
   ```bash
   # Should show the file exists
   dir models\dense_best.h5  # Windows
   ls models/dense_best.h5   # Linux/Mac
   ```

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
