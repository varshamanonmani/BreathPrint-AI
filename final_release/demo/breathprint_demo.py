import tkinter as tk
from tkinter import filedialog

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# =========================
# Model path
# =========================

MODEL_PATH = r"C:\BreathPrintFinal\final_release\model\breathprint_cycle_finetuned.keras"


# Classes

CLASSES = [
    "Both",
    "Crackle",
    "Normal",
    "Wheeze"
]


# =========================
# Load Model
# =========================

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded")


# =========================
# Prediction Function
# =========================

def predict_image():

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("PNG Images","*.png"),
            ("JPEG Images","*.jpg")
        ]
    )


    if not file_path:
        return


    img = image.load_img(
        file_path,
        target_size=(224,224)
    )


    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    prediction = model.predict(
        img_array
    )[0]


    result_index = np.argmax(
        prediction
    )


    result = CLASSES[result_index]


    output = f"""
BreathPrint Result

Prediction:
{result}


Confidence Scores:

Both     : {prediction[0]*100:.2f} %

Crackle  : {prediction[1]*100:.2f} %

Normal   : {prediction[2]*100:.2f} %

Wheeze   : {prediction[3]*100:.2f} %
"""


    result_text.config(
        text=output
    )



# =========================
# GUI
# =========================

window = tk.Tk()

window.title(
    "BreathPrint AI Respiratory Analysis"
)

window.geometry(
    "500x450"
)


title = tk.Label(
    window,
    text="BreathPrint AI",
    font=("Arial",22,"bold")
)

title.pack(
    pady=20
)


button = tk.Button(
    window,
    text="Select Lung Sound Spectrogram",
    command=predict_image,
    font=("Arial",12)
)

button.pack(
    pady=20
)


result_text = tk.Label(
    window,
    text="Waiting for input...",
    font=("Arial",12),
    justify="left"
)


result_text.pack(
    pady=20
)



window.mainloop()