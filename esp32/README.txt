BreathPrint ESP32 Module

Purpose:
Audio acquisition module for respiratory sound recording.

Hardware:
- ESP32-S3
- I2S MEMS Microphone
- OLED Display

Current Architecture:

Microphone
    |
    v
ESP32-S3
    |
    v
Bluetooth/WiFi
    |
    v
BreathPrint AI Model

AI Model:
breathprint.tflite

Inference:
External device (Laptop/Mobile)