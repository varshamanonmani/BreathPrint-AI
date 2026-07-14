import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


# Paths

RAW_PATH = r"C:\BreathPrintFinal\dataset\raw\ICBHI_final_database"

OUTPUT_PATH = r"C:\BreathPrintFinal\dataset\processed\cycle_spectrograms"


classes = [
    "normal",
    "crackle",
    "wheeze",
    "both"
]


for c in classes:
    os.makedirs(
        os.path.join(OUTPUT_PATH, c),
        exist_ok=True
    )


SAMPLE_RATE = 16000


def get_label(crackle, wheeze):

    if crackle == 1 and wheeze == 1:
        return "both"

    elif crackle == 1:
        return "crackle"

    elif wheeze == 1:
        return "wheeze"

    else:
        return "normal"



count = 0


for file in os.listdir(RAW_PATH):

    if file.endswith(".wav"):

        wav_path = os.path.join(
            RAW_PATH,
            file
        )

        txt_path = os.path.join(
            RAW_PATH,
            file.replace(".wav", ".txt")
        )


        if not os.path.exists(txt_path):
            continue


        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE
        )


        with open(txt_path, "r") as f:
            annotations = f.readlines()



        cycle_number = 0


        for line in annotations:

            start, end, crackle, wheeze = line.split()


            start = float(start)
            end = float(end)

            crackle = int(crackle)
            wheeze = int(wheeze)


            label = get_label(
                crackle,
                wheeze
            )


            start_sample = int(
                start * SAMPLE_RATE
            )

            end_sample = int(
                end * SAMPLE_RATE
            )


            cycle = audio[
                start_sample:end_sample
            ]


            # Remove very short cycles

            if len(cycle) < SAMPLE_RATE // 2:
                continue



            mel = librosa.feature.melspectrogram(
                y=cycle,
                sr=SAMPLE_RATE,
                n_mels=128
            )


            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )


            save_name = (
                file.replace(".wav", "")
                +
                "_cycle_"
                +
                str(cycle_number)
                +
                ".png"
            )


            save_path = os.path.join(
                OUTPUT_PATH,
                label,
                save_name
            )


            # Clean spectrogram image

            plt.figure(
                figsize=(2.56, 2.56),
                dpi=100
            )


            librosa.display.specshow(
                mel_db,
                sr=SAMPLE_RATE,
                cmap="magma"
            )


            plt.axis("off")

            plt.tight_layout(
                pad=0
            )


            plt.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0
            )


            plt.close()


            cycle_number += 1
            count += 1



print("\nCycle spectrogram generation completed!")
print("Total cycle spectrograms:", count)