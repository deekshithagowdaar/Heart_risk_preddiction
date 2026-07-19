import streamlit as st
import requests
from PIL import Image
import numpy as np
from gradcam import get_gradcam_image

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# =========================
# PDF FUNCTION
# =========================
def generate_pdf(predicted_class, confidence, age, bp, chol, health_risk, final_risk, final_label):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("AI Cardiovascular Risk Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Eye Condition: {predicted_class}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence:.2f}", styles['Normal']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Age: {age}", styles['Normal']))
    content.append(Paragraph(f"Blood Pressure: {bp}", styles['Normal']))
    content.append(Paragraph(f"Cholesterol: {chol}", styles['Normal']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Health Risk Score: {health_risk:.2f}", styles['Normal']))
    content.append(Paragraph(f"Final Risk Score: {final_risk:.2f}", styles['Normal']))
    content.append(Paragraph(f"Final Risk Level: {final_label}", styles['Normal']))

    doc.build(content)
    buffer.seek(0)

    return buffer


# =========================
# UI
# =========================
st.title("AI Cardiovascular Risk Prediction System")

st.sidebar.header("Enter Health Details")

age = st.sidebar.number_input("Age", 1, 100, 30)
bp = st.sidebar.number_input("Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):

        try:
            # ✅ CORRECT FILE SENDING (IMPORTANT FIX)
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),   # 🔥 THIS IS THE FIX
                    uploaded_file.type
                )
            }

            data = {
                "age": age,
                "bp": bp,
                "chol": chol
            }

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files,
                data=data
            )

            # ✅ DEBUG IF ERROR
            if response.status_code != 200:
                st.error("❌ Backend Error")
                st.text(response.text)
                st.stop()

            result = response.json()

        except Exception as e:
            st.error("⚠️ Backend not running or crashed")
            st.text(str(e))
            st.stop()

        # =========================
        # OUTPUT
        # =========================
        predicted_class = result["eye_condition"]
        confidence = result["confidence"]
        health_risk = result["health_risk"]
        final_risk = result["final_risk"]
        final_label = result["risk_level"]

        st.subheader("Results")

        st.write(f"**Eye Condition:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**Health Risk Score:** {health_risk:.2f}")
        st.write(f"**Final Risk Score:** {final_risk:.2f}")

        st.success(f"Final Risk Level: {final_label}")

        # =========================
        # GRAD-CAM
        # =========================
        st.subheader("Grad-CAM Explanation 🔥")

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        gradcam_img = get_gradcam_image(img_array)
        st.image(gradcam_img, caption="Model Attention Heatmap")

        # =========================
        # PDF DOWNLOAD
        # =========================
        st.subheader("Download Report 📄")

        pdf = generate_pdf(
            predicted_class,
            confidence,
            age,
            bp,
            chol,
            health_risk,
            final_risk,
            final_label
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf,
            file_name="cardio_risk_report.pdf",
            mime="application/pdf"
        )