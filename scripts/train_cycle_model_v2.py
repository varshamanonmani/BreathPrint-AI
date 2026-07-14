import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.utils.class_weight import compute_class_weight


# ===============================
# Paths
# ===============================

DATASET_PATH = r"C:\BreathPrintFinal\dataset\processed\cycle_spectrograms"

MODEL_PATH = r"C:\BreathPrintFinal\models\breathprint_cycle_mobilenetv2.keras"


# ===============================
# Parameters
# ===============================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 30


# ===============================
# Data generators
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
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


print("\nClass Mapping:")
print(train_data.class_indices)



# ===============================
# Class weights
# ===============================

labels = train_data.classes

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)


class_weights = dict(
    enumerate(weights)
)


print("\nClass Weights:")
print(class_weights)



# ===============================
# MobileNetV2 model
# ===============================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)


base_model.trainable = False



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.4)(x)

x = Dense(
    128,
    activation="relu"
)(x)


x = Dropout(0.3)(x)


output = Dense(
    4,
    activation="softmax"
)(x)



model = Model(
    inputs=base_model.input,
    outputs=output
)



model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)



model.summary()



# ===============================
# Training
# ===============================

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    class_weight=class_weights

)



# ===============================
# Save model
# ===============================

model.save(
    MODEL_PATH
)


print("\nBreathPrint Cycle MobileNetV2 Training Completed!")
print("Saved:", MODEL_PATH)