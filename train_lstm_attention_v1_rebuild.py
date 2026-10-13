import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SEED
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "lstm_features"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATA_DIR,
    "validation"
)

TEST_DIR = os.path.join(
    DATA_DIR,
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "lstm_v1"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
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

NUM_CLASSES = len(
    EMOTIONS
)


# ============================================================
# DATA SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20

FEATURE_DIM = 256


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32

EPOCHS = 40

LEARNING_RATE = 1e-3


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_final.keras"
)

CLASS_FILE_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_classes.json"
)

MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_mean.npy"
)

STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_std.npy"
)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print("")
    print("=" * 70)
    print("CHECKING LSTM V1 DATASET")
    print("=" * 70)

    ok = True

    for split_name, split_path in [

        ("TRAIN", TRAIN_DIR),

        ("VALIDATION", VAL_DIR),

        ("TEST", TEST_DIR)

    ]:

        print("")
        print(split_name)
        print("-" * 40)

        total = 0

        if not os.path.isdir(
            split_path
        ):

            print(
                "[ERROR] Missing split:"
            )

            print(
                split_path
            )

            ok = False

            continue


        for emotion in EMOTIONS:

            folder = os.path.join(
                split_path,
                emotion
            )

            if not os.path.isdir(
                folder
            ):

                print(
                    "[ERROR] Missing emotion:",
                    emotion
                )

                ok = False

                continue


            count = len([
                f
                for f in os.listdir(folder)
                if f.lower().endswith(".npy")
            ])


            total += count


            print(
                f"{emotion:10s}: {count}"
            )


        print(
            f"{'TOTAL':10s}: {total}"
        )


    return ok


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(
    split_dir
):

    x_data = []
    y_data = []


    for class_index, emotion in enumerate(
        EMOTIONS
    ):

        emotion_dir = os.path.join(
            split_dir,
            emotion
        )


        files = sorted([
            f
            for f in os.listdir(
                emotion_dir
            )
            if f.lower().endswith(
                ".npy"
            )
        ])


        for filename in files:

            path = os.path.join(
                emotion_dir,
                filename
            )


            features = np.load(
                path
            )


            if features.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIM
            ):

                raise ValueError(
                    f"Invalid shape: "
                    f"{features.shape}\n"
                    f"File: {path}"
                )


            x_data.append(
                features.astype(
                    np.float32
                )
            )


            y_data.append(
                class_index
            )


    x_data = np.asarray(
        x_data,
        dtype=np.float32
    )


    y_data = np.asarray(
        y_data,
        dtype=np.int32
    )


    return (
        x_data,
        y_data
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(
    x_train,
    x_val,
    x_test
):

    print("")
    print(
        "[INFO] Calculating normalization "
        "from TRAIN only..."
    )


    feature_mean = np.mean(
        x_train,
        axis=(0, 1)
    )


    feature_std = np.std(
        x_train,
        axis=(0, 1)
    )


    feature_std = np.where(
        feature_std < 1e-6,
        1e-6,
        feature_std
    )


    x_train = (
        x_train - feature_mean
    ) / feature_std


    x_val = (
        x_val - feature_mean
    ) / feature_std


    x_test = (
        x_test - feature_mean
    ) / feature_std


    np.save(
        MEAN_PATH,
        feature_mean
    )


    np.save(
        STD_PATH,
        feature_std
    )


    print(
        "[SUCCESS] V1 normalization saved."
    )


    return (
        x_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32)
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_class_weights(
    y_train
):

    weights = compute_class_weight(

        class_weight="balanced",

        classes=np.arange(
            NUM_CLASSES
        ),

        y=y_train

    )


    class_weights = {

        index:
            float(weight)

        for index, weight
        in enumerate(weights)

    }


    print("")
    print("CLASS WEIGHTS")
    print("-" * 40)


    for index, weight in class_weights.items():

        print(
            f"{EMOTIONS[index]:10s}: "
            f"{weight:.4f}"
        )


    return class_weights


# ============================================================
# TEMPORAL ATTENTION
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="EmotionAI"
)
class TemporalAttention(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        units=64,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.units = units


        self.score_dense = (
            tf.keras.layers.Dense(
                units,
                activation="tanh"
            )
        )


        self.weight_dense = (
            tf.keras.layers.Dense(
                1
            )
        )


    def call(
        self,
        inputs
    ):

        scores = self.score_dense(
            inputs
        )


        scores = self.weight_dense(
            scores
        )


        weights = tf.nn.softmax(
            scores,
            axis=1
        )


        context = tf.reduce_sum(
            weights * inputs,
            axis=1
        )


        return context


    def get_config(
        self
    ):

        config = super().get_config()


        config.update({
            "units": self.units
        })


        return config


# ============================================================
# BUILD EXACT V1 MODEL
# ============================================================

def build_model():

    print("")
    print("=" * 70)
    print("BUILDING ORIGINAL V1")
    print("=" * 70)


    inputs = tf.keras.Input(

        shape=(
            SEQUENCE_LENGTH,
            FEATURE_DIM
        ),

        name="sequence_features"

    )


    # --------------------------------------------------------
    # Mild feature noise
    # --------------------------------------------------------

    x = tf.keras.layers.GaussianNoise(
        0.02
    )(inputs)


    # --------------------------------------------------------
    # BiLSTM 1
    # --------------------------------------------------------

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            128,

            return_sequences=True,

            dropout=0.25,

            recurrent_dropout=0.15

        )

    )(x)


    # --------------------------------------------------------
    # BiLSTM 2
    # --------------------------------------------------------

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            64,

            return_sequences=True,

            dropout=0.25,

            recurrent_dropout=0.15

        )

    )(x)


    # --------------------------------------------------------
    # Temporal normalization
    # --------------------------------------------------------

    x = tf.keras.layers.LayerNormalization()(
        x
    )


    # --------------------------------------------------------
    # Original attention
    # --------------------------------------------------------

    context = TemporalAttention(

        units=64,

        name="temporal_attention"

    )(x)


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    x = tf.keras.layers.Dropout(
        0.35
    )(context)


    x = tf.keras.layers.Dense(

        128,

        activation="relu",

        kernel_regularizer=(
            tf.keras.regularizers.l2(
                1e-4
            )
        )

    )(x)


    x = tf.keras.layers.BatchNormalization()(
        x
    )


    x = tf.keras.layers.Dropout(
        0.35
    )(x)


    outputs = tf.keras.layers.Dense(

        NUM_CLASSES,

        activation="softmax",

        name="emotion_output"

    )(x)


    model = tf.keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="BiLSTM_Attention_V1_Rebuild"

    )


    return model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks():

    return [

        tf.keras.callbacks.ModelCheckpoint(

            BEST_MODEL_PATH,

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            verbose=1

        ),


        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            mode="min",

            patience=6,

            restore_best_weights=True,

            verbose=1

        ),


        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            mode="min",

            factor=0.4,

            patience=2,

            min_lr=1e-6,

            verbose=1

        )

    ]


# ============================================================
# GRAPHS
# ============================================================

def save_history(
    history
):

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )


    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        "LSTM Attention V1 Accuracy"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(
            RESULT_DIR,
            "lstm_attention_v1_accuracy.png"
        ),

        dpi=200

    )


    plt.close()


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )


    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "LSTM Attention V1 Loss"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(
            RESULT_DIR,
            "lstm_attention_v1_loss.png"
        ),

        dpi=200

    )


    plt.close()


# ============================================================
# TEST
# ============================================================

def evaluate_model(
    model,
    x_test,
    y_test
):

    print("")
    print("=" * 70)
    print("FINAL TEST - V1 REBUILD - UNSEEN ACTORS")
    print("=" * 70)


    probabilities = model.predict(

        x_test,

        batch_size=BATCH_SIZE,

        verbose=0

    )


    y_pred = np.argmax(

        probabilities,

        axis=1

    )


    accuracy = accuracy_score(

        y_test,

        y_pred

    )


    macro_f1 = f1_score(

        y_test,

        y_pred,

        average="macro",

        zero_division=0

    )


    weighted_f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    print("")
    print(
        f"Test Accuracy : {accuracy:.4f}"
    )


    print(
        f"Macro F1      : {macro_f1:.4f}"
    )


    print(
        f"Weighted F1   : {weighted_f1:.4f}"
    )


    report = classification_report(

        y_test,

        y_pred,

        target_names=EMOTIONS,

        zero_division=0

    )


    print("")
    print(
        report
    )


    report_path = os.path.join(

        RESULT_DIR,

        "lstm_attention_v1_test_report.txt"

    )


    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Macro F1: {macro_f1:.6f}\n"
        )

        file.write(
            f"Weighted F1: {weighted_f1:.6f}\n\n"
        )

        file.write(
            report
        )


    matrix = confusion_matrix(

        y_test,

        y_pred

    )


    plt.figure(
        figsize=(9, 8)
    )


    plt.imshow(
        matrix
    )


    plt.title(
        "LSTM Attention V1 Confusion Matrix"
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Actual"
    )


    plt.xticks(

        range(
            NUM_CLASSES
        ),

        EMOTIONS,

        rotation=45

    )


    plt.yticks(

        range(
            NUM_CLASSES
        ),

        EMOTIONS

    )


    for row in range(
        matrix.shape[0]
    ):

        for col in range(
            matrix.shape[1]
        ):

            plt.text(

                col,

                row,

                str(
                    matrix[
                        row,
                        col
                    ]
                ),

                ha="center",

                va="center"

            )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "lstm_attention_v1_confusion_matrix.png"

        ),

        dpi=200

    )


    plt.close()


    return (
        accuracy,
        macro_f1,
        weighted_f1
    )


# ============================================================
# MAIN
# ============================================================

def train():

    print("")
    print("=" * 70)
    print("REBUILDING ORIGINAL LSTM + ATTENTION V1")
    print("=" * 70)


    if not check_dataset():

        print(
            "[STOP] Dataset check failed."
        )

        return


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("")
    print(
        "[INFO] Loading Train..."
    )

    x_train, y_train = load_split(
        TRAIN_DIR
    )


    print(
        "[INFO] Loading Validation..."
    )

    x_val, y_val = load_split(
        VAL_DIR
    )


    print(
        "[INFO] Loading Test..."
    )

    x_test, y_test = load_split(
        TEST_DIR
    )


    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    print("")
    print("DATA SHAPES")
    print("-" * 40)


    print(
        "Train:",
        x_train.shape,
        y_train.shape
    )


    print(
        "Validation:",
        x_val.shape,
        y_val.shape
    )


    print(
        "Test:",
        x_test.shape,
        y_test.shape
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    (
        x_train,
        x_val,
        x_test

    ) = normalize_features(

        x_train,
        x_val,
        x_test

    )


    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = get_class_weights(
        y_train
    )


    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    model = build_model()


    model.compile(

        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            )
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]

    )


    model.summary()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(

        x_train,

        y_train,

        validation_data=(
            x_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=True,

        class_weight=class_weights,

        callbacks=get_callbacks(),

        verbose=1

    )


    # --------------------------------------------------------
    # Save restored-best model
    # --------------------------------------------------------

    model.save(
        FINAL_MODEL_PATH
    )


    # --------------------------------------------------------
    # Save classes
    # --------------------------------------------------------

    with open(

        CLASS_FILE_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            EMOTIONS,

            file,

            indent=4

        )


    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    save_history(
        history
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    (
        accuracy,
        macro_f1,
        weighted_f1

    ) = evaluate_model(

        model,

        x_test,

        y_test

    )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("V1 REBUILD COMPLETED")
    print("=" * 70)


    print("")
    print(
        "Best checkpoint:"
    )

    print(
        BEST_MODEL_PATH
    )


    print("")
    print(
        "Final restored model:"
    )

    print(
        FINAL_MODEL_PATH
    )


    print("")
    print(
        f"Accuracy    : {accuracy:.4f}"
    )


    print(
        f"Macro F1    : {macro_f1:.4f}"
    )


    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train()