import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("models/image_model.h5")

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# =========================
# RISK MAPPING
# =========================
risk_map = {
    "normal": 0.1,
    "cataract": 0.4,
    "glaucoma": 0.7,
    "diabetic_retinopathy": 0.9
}

# =========================
# USER INPUT (HEALTH DATA)
# =========================
print("\nEnter Patient Details:")

age = int(input("Age: "))
bp = int(input("Blood Pressure: "))
cholesterol = int(input("Cholesterol: "))

# =========================
# LOAD IMAGE
# =========================
img_path = input("\nEnter image path (e.g., test.jpg): ")

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# =========================
# IMAGE PREDICTION
# =========================
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

# =========================
# IMAGE RISK
# =========================
risk_score = risk_map[predicted_class]

if risk_score < 0.3:
    risk_label = "Low Risk"
elif risk_score < 0.7:
    risk_label = "Moderate Risk"
else:
    risk_label = "High Risk"

# =========================
# HEALTH RISK
# =========================
age_score = age / 100
bp_score = bp / 200
chol_score = cholesterol / 300

health_risk = (age_score + bp_score + chol_score) / 3

# =========================
# FINAL MULTIMODAL RISK
# =========================
final_risk = (risk_score * 0.6) + (health_risk * 0.4)

if final_risk < 0.3:
    final_label = "Low Risk"
elif final_risk < 0.7:
    final_label = "Moderate Risk"
else:
    final_label = "High Risk"

# =========================
# OUTPUT
# =========================
print("\n----- IMAGE ANALYSIS -----")
print(f"Eye Condition: {predicted_class}")
print(f"Confidence: {confidence:.2f}")
print(f"Image Risk Score: {risk_score}")
print(f"Image Risk Level: {risk_label}")

print("\n----- HEALTH DATA -----")
print(f"Age: {age}")
print(f"Blood Pressure: {bp}")
print(f"Cholesterol: {cholesterol}")
print(f"Health Risk Score: {health_risk:.2f}")

print("\n----- FINAL RESULT -----")
print(f"Final Risk Score: {final_risk:.2f}")
print(f"Final Risk Level: {final_label}")