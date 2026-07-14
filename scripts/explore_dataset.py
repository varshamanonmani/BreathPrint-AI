import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = r"C:\BreathPrintFinal\dataset\raw\ICBHI_final_database"

# Find audio files
audio_files = [
    f for f in os.listdir(DATASET_PATH)
    if f.endswith(".wav")
]

print("Total lung recordings found:", len(audio_files))

# Select first recording
sample_file = audio_files[0]
audio_path = os.path.join(DATASET_PATH, sample_file)

print("Sample file:", sample_file)

# Load audio
audio, sr = librosa.load(audio_path, sr=None)

print("Sampling rate:", sr)
print("Audio duration:", round(len(audio)/sr, 2), "seconds")


# Plot waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title("BreathPrint - Lung Sound Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()


# Generate spectrogram
spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128
)

spectrogram_db = librosa.power_to_db(
    spectrogram,
    ref=1.0
)

plt.figure(figsize=(12, 4))
librosa.display.specshow(
    spectrogram_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar()
plt.title("BreathPrint - Lung Sound Mel Spectrogram")
plt.show()