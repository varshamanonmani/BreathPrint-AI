import os
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# Paths

DATASET_PATH = r"C:\BreathPrintFinal\dataset\processed\cycle_spectrograms"

MODEL_PATH = r"C:\BreathPrintFinal\models\breathprint_cycle_cnn.keras"


IMG_SIZE = (128,128)
BATCH_SIZE = 32


# Data generators

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)


val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("\nClass mapping:")
print(train_data.class_indices)


# Class weights

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(weights))

print("\nClass weights:")
print(class_weights)



# CNN model

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    ),

    BatchNormalization(),

    MaxPooling2D(2,2),


    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(2,2),


    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    BatchNormalization(),

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


early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=4
)



history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)


print("\nBreathPrint Cycle CNN Training Completed!")