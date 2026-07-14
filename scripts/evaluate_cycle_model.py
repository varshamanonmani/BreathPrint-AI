import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================
# Paths
# ============================

DATASET_PATH = r"C:\BreathPrintFinal\dataset\processed\cycle_spectrograms"

MODEL_PATH = r"C:\BreathPrintFinal\models\breathprint_cycle_finetuned.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================
# Load Model
# ============================

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================
# Validation Dataset
# ============================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ============================
# Predictions
# ============================

print("\nRunning predictions...\n")

predictions = model.predict(
    val_data,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

class_names = list(val_data.class_indices.keys())

# ============================
# Accuracy
# ============================

accuracy = accuracy_score(y_true, y_pred)

print("=" * 60)
print(f"Validation Accuracy : {accuracy * 100:.2f}%")
print("=" * 60)

# ============================
# Class Names
# ============================

print("\nClass Names:")
print(class_names)

# ============================
# Classification Report
# ============================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# ============================
# Confusion Matrix
# ============================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

print("\nEvaluation Completed Successfully!")