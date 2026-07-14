import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Paths
DATASET_PATH = r"C:\BreathPrintFinal\dataset\raw\ICBHI_final_database"

OUTPUT_PATH = r"C:\BreathPrintFinal\dataset\processed\spectrograms"


# Label folders
LABELS = {
    "normal": 0,
    "crackle": 1,
    "wheeze": 2,
    "both": 3
}


def get_label(annotation_file):

    crackle = False
    wheeze = False

    with open(annotation_file, "r") as file:
        for line in file:
            values = line.split()

            if len(values) == 4:
                if values[2] == "1":
                    crackle = True

                if values[3] == "1":
                    wheeze = True


    if crackle and wheeze:
        return "both"

    elif crackle:
        return "crackle"

    elif wheeze:
        return "wheeze"

    else:
        return "normal"



def create_spectrogram(audio_file):

    audio_path = os.path.join(DATASET_PATH, audio_file)

    audio, sr = librosa.load(
        audio_path,
        sr=4000
    )


    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )


    mel_db = librosa.power_to_db(
        mel,
        ref=1.0
    )


    return mel_db



wav_files = [
    file for file in os.listdir(DATASET_PATH)
    if file.endswith(".wav")
]


print("Total files:", len(wav_files))


for wav in tqdm(wav_files):

    txt_file = wav.replace(".wav", ".txt")

    annotation_path = os.path.join(
        DATASET_PATH,
        txt_file
    )


    label = get_label(annotation_path)


    spectrogram = create_spectrogram(wav)


    save_path = os.path.join(
        OUTPUT_PATH,
        label,
        wav.replace(".wav", ".png")
    )


    plt.imsave(
        save_path,
        spectrogram
    )


print("Spectrogram generation completed!")