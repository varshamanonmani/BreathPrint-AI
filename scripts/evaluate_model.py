import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODEL_PATH = r"C:\BreathPrintFinal\models\breathprint_cnn.h5"

DATASET_PATH = r"C:\BreathPrintFinal\dataset\processed\spectrograms"


IMG_SIZE = (128,128)
BATCH_SIZE = 16


# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")


datagen = ImageDataGenerator(
    rescale=1./255
)


test_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


predictions = model.predict(test_data)


y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes


labels = list(test_data.class_indices.keys())


print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=labels
    )
)


print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)