import os
import shutil
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from train_lstm_attention_v1_rebuild import (
    TemporalAttention
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "lstm_features",
    "validation"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

BEST_CHECKPOINT = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_best.keras"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_final.keras"
)

SELECTED_MODEL = os.path.join(
    MODEL_DIR,
    "emotion_temporal_best.keras"
)

MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_mean.npy"
)

STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_std.npy"
)


EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def load_validation():

    x = []
    y = []

    for class_index, emotion in enumerate(
        EMOTIONS
    ):

        folder = os.path.join(
            VAL_DIR,
            emotion
        )

        for filename in sorted(
            os.listdir(folder)
        ):

            if not filename.endswith(".npy"):
                continue

            path = os.path.join(
                folder,
                filename
            )

            features = np.load(
                path
            ).astype(
                np.float32
            )

            x.append(
                features
            )

            y.append(
                class_index
            )


    return (
        np.asarray(
            x,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.int32
        )
    )


def evaluate_model(
    model_path,
    x_val,
    y_val
):

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "TemporalAttention":
                TemporalAttention
        },
        compile=False
    )


    probabilities = model.predict(
        x_val,
        batch_size=32,
        verbose=0
    )


    predictions = np.argmax(
        probabilities,
        axis=1
    )


    accuracy = accuracy_score(
        y_val,
        predictions
    )


    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )


    return (
        accuracy,
        macro_f1
    )


def main():

    print("")
    print("=" * 70)
    print("SELECTING BEST V1 MODEL")
    print("=" * 70)


    x_val, y_val = load_validation()


    mean = np.load(
        MEAN_PATH
    )

    std = np.load(
        STD_PATH
    )


    x_val = (
        x_val - mean
    ) / std


    print("")
    print(
        "Validation shape:",
        x_val.shape
    )


    best_accuracy, best_f1 = evaluate_model(
        BEST_CHECKPOINT,
        x_val,
        y_val
    )


    final_accuracy, final_f1 = evaluate_model(
        FINAL_MODEL,
        x_val,
        y_val
    )


    print("")
    print(
        "CHECKPOINT MODEL"
    )

    print(
        f"Accuracy : {best_accuracy:.4f}"
    )

    print(
        f"Macro F1 : {best_f1:.4f}"
    )


    print("")
    print(
        "FINAL RESTORED MODEL"
    )

    print(
        f"Accuracy : {final_accuracy:.4f}"
    )

    print(
        f"Macro F1 : {final_f1:.4f}"
    )


    # انتخاب فقط با Validation Macro F1

    if best_f1 >= final_f1:

        winner = BEST_CHECKPOINT

        print("")
        print(
            "[WINNER] Best checkpoint"
        )

    else:

        winner = FINAL_MODEL

        print("")
        print(
            "[WINNER] Final restored model"
        )


    shutil.copy2(
        winner,
        SELECTED_MODEL
    )


    print("")
    print(
        "[SUCCESS] Selected model saved as:"
    )

    print(
        SELECTED_MODEL
    )


if __name__ == "__main__":

    main()