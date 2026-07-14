import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


# ==============================
# Paths
# ==============================

MODEL_PATH = r"C:\BreathPrintFinal\final_release\model\breathprint_cycle_finetuned.keras"


# Put your test spectrogram path here
IMAGE_PATH = r"C:\BreathPrintFinal\dataset\processed\cycle_spectrograms\normal\101_1b1_Al_sc_Meditron_cycle_0.png"


# ==============================
# Class names
# ==============================

CLASS_NAMES = [
    "both",
    "crackle",
    "normal",
    "wheeze"
]


# ==============================
# Load model
# ==============================

print("Loading BreathPrint model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==============================
# Load image
# ==============================

img = image.load_img(
    IMAGE_PATH,
    target_size=(224,224)
)


img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ==============================
# Prediction
# ==============================

prediction = model.predict(img_array)


probabilities = prediction[0]


predicted_index = np.argmax(probabilities)


predicted_class = CLASS_NAMES[predicted_index]


confidence = probabilities[predicted_index] * 100


# ==============================
# Display Result
# ==============================

print("\n==============================")
print("      BreathPrint Result")
print("==============================")

print("\nPrediction:")
print(predicted_class.upper())


print("\nConfidence Scores:")

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name}: {probabilities[i]*100:.2f}%"
    )


print("\nFinal Confidence:")
print(f"{confidence:.2f}%")

print("==============================")