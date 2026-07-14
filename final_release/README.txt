=========================================
BreathPrint
Respiratory Disorder Pre-Screening System
=========================================


Project Overview:

BreathPrint is an AI-based respiratory sound
analysis system that classifies lung sounds
into four categories:

1. Normal
2. Crackle
3. Wheeze
4. Both


Dataset:

ICBHI 2017 Respiratory Sound Database


AI Model:

MobileNetV2 Transfer Learning Model

Input:
Respiratory cycle spectrogram

Output:
Respiratory sound classification


Software:

Python
TensorFlow
Keras


Deployment Model:

TensorFlow Lite model available for
mobile/embedded deployment.


Current Prototype:

Audio acquisition:
ESP32-S3 + I2S MEMS microphone

AI processing:
External device using BreathPrint model


Future Improvements:

- Larger respiratory datasets
- Real-time audio classification
- Edge AI deployment
- Clinical validation

=========================================