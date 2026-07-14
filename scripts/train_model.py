import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


# Dataset path
DATASET_PATH = r"C:\BreathPrintFinal\dataset\processed\spectrograms"

MODEL_PATH = r"C:\BreathPrintFinal\models\breathprint_cnn.h5"


# Image settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 16


# Data preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


print("Class mapping:")
print(train_data.class_indices)


# CNN Model

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    ),

    MaxPooling2D(2,2),


    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),


    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),


    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),


    Dense(
        4,
        activation="softmax"
    )

])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True
)


history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=25,
    callbacks=[checkpoint]
)


print("Training completed!")