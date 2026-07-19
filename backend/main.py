from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2
import os
from fpdf import FPDF

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# =========================
# CREATE APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STATIC FOLDER
# =========================
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("models/image_model.h5")

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

risk_map = {
    "normal": 0.1,
    "cataract": 0.4,
    "glaucoma": 0.7,
    "diabetic_retinopathy": 0.9
}

# =========================
# PREPROCESSING
# =========================
def preprocess_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    h, w, _ = img.shape
    crop = img[h//8:7*h//8, w//8:7*w//8]

    crop = cv2.resize(crop, (224, 224))

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    crop = cv2.merge((l, a, b))
    crop = cv2.cvtColor(crop, cv2.COLOR_LAB2BGR)

    crop = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    crop = cv2.filter2D(crop, -1, kernel)
    crop = crop / 255.0

    return np.expand_dims(crop, axis=0)

# =========================
# PDF REPORT
# =========================
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="CardioVision AI Report", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Eye Condition: {data['eye_condition']}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {data['confidence']}", ln=True)
    pdf.cell(200, 10, txt=f"Health Risk: {data['health_risk']}", ln=True)
    pdf.cell(200, 10, txt=f"Final Risk: {data['final_risk']}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {data['risk_level']}", ln=True)

    pdf.output("static/report.pdf")

# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"message": "API running 🚀"}

# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    bp: int = Form(...),
    chol: int = Form(...)
):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        img_array = preprocess_image(img)

        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        risk_score = risk_map[predicted_class]

        age_score = age / 100
        bp_score = bp / 200
        chol_score = chol / 300

        health_risk = (age_score + bp_score + chol_score) / 3

        final_risk = (risk_score * 0.6) + (health_risk * 0.4)

        if final_risk < 0.3:
            final_label = "Low Risk"
        elif final_risk < 0.7:
            final_label = "Moderate Risk"
        else:
            final_label = "High Risk"

        # Save uploaded image as fake heatmap for now
        img.save("static/heatmap.png")

        data = {
            "eye_condition": predicted_class,
            "confidence": round(confidence, 2),
            "health_risk": round(health_risk, 2),
            "final_risk": round(final_risk, 2),
            "risk_level": final_label
        }

        generate_pdf(data)

        data["heatmap_url"] = "http://127.0.0.1:8000/static/heatmap.png"
        data["pdf_url"] = "http://127.0.0.1:8000/static/report.pdf"

        return data

    except Exception as e:
        return {"error": str(e)}