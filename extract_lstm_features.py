import os
import json
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ravdess_sequences"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "lstm_features"
)

FEATURE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_feature_extractor.keras"
)


# ============================================================
# CLASSES
# ============================================================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ============================================================
# ACTOR SPLIT
# ============================================================

def get_split(actor_id):

    number = int(actor_id)

    if 1 <= number <= 16:
        return "train"

    if 17 <= number <= 20:
        return "validation"

    if 21 <= number <= 24:
        return "test"

    return None


# ============================================================
# CREATE FOLDERS
# ============================================================

for split in [
    "train",
    "validation",
    "test"
]:

    for emotion in EMOTIONS:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                emotion
            ),
            exist_ok=True
        )


# ============================================================
# LOAD CNN FEATURE EXTRACTOR
# ============================================================

print("")
print("=" * 70)
print("LOADING CNN FEATURE EXTRACTOR")
print("=" * 70)

feature_model = tf.keras.models.load_model(
    FEATURE_MODEL_PATH,
    compile=False
)

print(
    "[SUCCESS] CNN feature extractor loaded."
)


# ============================================================
# EXTRACT
# ============================================================

counters = {
    "train": 0,
    "validation": 0,
    "test": 0
}

broken = 0


for actor_folder in sorted(
    os.listdir(SEQUENCE_DIR)
):

    actor_path = os.path.join(
        SEQUENCE_DIR,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    if not actor_folder.startswith(
        "actor_"
    ):
        continue

    actor_id = actor_folder.replace(
        "actor_",
        ""
    )

    split = get_split(
        actor_id
    )

    if split is None:
        continue

    print(
        f"[ACTOR {actor_id}] -> {split}"
    )


    for emotion in EMOTIONS:

        emotion_path = os.path.join(
            actor_path,
            emotion
        )

        if not os.path.isdir(
            emotion_path
        ):
            continue


        for filename in sorted(
            os.listdir(emotion_path)
        ):

            if not filename.endswith(
                ".npy"
            ):
                continue


            source_path = os.path.join(
                emotion_path,
                filename
            )


            try:

                sequence = np.load(
                    source_path
                )

                # Expected:
                # 20 × 160 × 160 × 3

                if sequence.shape != (
                    20,
                    160,
                    160,
                    3
                ):

                    print(
                        "[INVALID SHAPE]",
                        filename,
                        sequence.shape
                    )

                    broken += 1

                    continue


                sequence = sequence.astype(
                    np.float32
                )


                # CNN receives all 20 frames
                features = feature_model.predict(
                    sequence,
                    batch_size=20,
                    verbose=0
                )


                # Expected:
                # 20 × 256

                if features.shape != (
                    20,
                    256
                ):

                    print(
                        "[INVALID FEATURE SHAPE]",
                        features.shape
                    )

                    broken += 1

                    continue


                output_path = os.path.join(
                    OUTPUT_DIR,
                    split,
                    emotion,
                    filename
                )


                np.save(
                    output_path,
                    features
                )


                counters[
                    split
                ] += 1


            except Exception as error:

                print(
                    "[ERROR]",
                    filename,
                    error
                )

                broken += 1


# ============================================================
# REPORT
# ============================================================

print("")
print("=" * 70)
print("CNN FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print(
    "Train sequences      :",
    counters["train"]
)

print(
    "Validation sequences :",
    counters["validation"]
)

print(
    "Test sequences       :",
    counters["test"]
)

print(
    "Broken sequences     :",
    broken
)

print("")
print(
    "Each sample shape = (20, 256)"
)

print("")
print(
    "Saved to:"
)

print(
    OUTPUT_DIR
)