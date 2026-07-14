import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


DATASET_PATH = r"C:\BreathPrintFinal\dataset\processed\cycle_spectrograms"

MODEL_PATH = r"C:\BreathPrintFinal\models\breathprint_cycle_mobilenetv2.keras"

SAVE_PATH = r"C:\BreathPrintFinal\models\breathprint_cycle_finetuned.keras"


IMG_SIZE = (224,224)
BATCH_SIZE = 32


# Load existing model

model = load_model(MODEL_PATH)

print("Loaded BreathPrint MobileNetV2 model")


# Freeze all layers first

for layer in model.layers:
    layer.trainable = False


# Unfreeze last 40 layers

for layer in model.layers[-40:]:
    layer.trainable = True


print("\nTrainable layers:")
for layer in model.layers:
    if layer.trainable:
        print(layer.name)



# Data augmentation

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
)



train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)



# Compile with very low learning rate

model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)



callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3
    ),

    ModelCheckpoint(
        SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]



history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=25,

    callbacks=callbacks

)



print("\n================================")
print("Fine tuning completed!")
print("Saved model:")
print(SAVE_PATH)
print("================================")