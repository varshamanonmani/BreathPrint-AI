import tensorflow as tf


MODEL_PATH = r"C:\BreathPrintFinal\final_release\model\breathprint_cycle_finetuned.keras"

OUTPUT_PATH = r"C:\BreathPrintFinal\final_release\model\breathprint.tflite"


print("Loading Keras model...")

model = tf.keras.models.load_model(MODEL_PATH)


print("Converting to TensorFlow Lite...")


converter = tf.lite.TFLiteConverter.from_keras_model(model)


# Optimization for mobile/embedded deployment
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]


tflite_model = converter.convert()


with open(
    OUTPUT_PATH,
    "wb"
) as f:
    f.write(tflite_model)


print("==============================")
print("Conversion Completed!")
print("==============================")

print("Saved:")
print(OUTPUT_PATH)