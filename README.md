\# BreathPrint AI

\## Respiratory Disorder Pre-Screening Using Lung Sound Analysis





\## Overview



BreathPrint is an AI-powered respiratory screening system that analyzes lung sound recordings and classifies respiratory conditions using deep learning.



The system converts respiratory audio signals into spectrogram images and uses a MobileNetV2-based CNN model for classification.





\## Problem Statement



Respiratory diseases require early detection, especially in rural and low-resource healthcare environments.



BreathPrint aims to provide a low-cost AI-assisted screening approach.





\## Classes Detected



The model classifies respiratory sounds into:



\- Normal

\- Crackle

\- Wheeze

\- Both (Crackle + Wheeze)





\## System Architecture



Audio Input  

↓  

Respiratory Sound Processing  

↓  

Spectrogram Generation  

↓  

MobileNetV2 Deep Learning Model  

↓  

Disease Classification  

↓  

Prediction \& Confidence Score





\## Technologies Used



\### Artificial Intelligence

\- Python

\- TensorFlow

\- Keras

\- MobileNetV2 Transfer Learning



\### Signal Processing

\- Spectrogram generation

\- Respiratory cycle segmentation



\### Deployment

\- TensorFlow Lite

\- ESP32-S3 (Prototype)





\## Model Performance



Validation Accuracy:

\~59%



Demo Prediction Confidence:

90%+ for sample cases





\## Project Structure



