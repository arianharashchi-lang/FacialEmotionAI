import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# PROJECT PATHS
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
    "experiments",
    "v3"
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
# EMOTION CLASSES
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
# FEATURE SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20

FEATURE_DIM = 256


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 3e-4


# ============================================================
# OUTPUT PATHS
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v3_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v3_final.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v3_classes.json"
)

MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v3_feature_mean.npy"
)

STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v3_feature_std.npy"
)

REPORT_PATH = os.path.join(
    RESULT_DIR,
    "lstm_attention_v3_test_report.txt"
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "lstm_attention_v3_metrics.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULT_DIR,
    "lstm_attention_v3_confusion_matrix.png"
)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print("")
    print("=" * 70)
    print("CHECKING LSTM V3 DATASET")
    print("=" * 70)

    dataset_ok = True

    for split_name, split_dir in [

        ("TRAIN", TRAIN_DIR),

        ("VALIDATION", VAL_DIR),

        ("TEST", TEST_DIR)

    ]:

        print("")
        print(split_name)
        print("-" * 40)

        split_total = 0

        if not os.path.isdir(
            split_dir
        ):

            print(
                "[ERROR] Folder not found:"
            )

            print(
                split_dir
            )

            dataset_ok = False

            continue


        for emotion in EMOTIONS:

            emotion_dir = os.path.join(
                split_dir,
                emotion
            )

            if not os.path.isdir(
                emotion_dir
            ):

                print(
                    f"[ERROR] Missing emotion folder: "
                    f"{emotion}"
                )

                dataset_ok = False

                continue


            files = [

                file

                for file in os.listdir(
                    emotion_dir
                )

                if file.lower().endswith(
                    ".npy"
                )

            ]


            count = len(
                files
            )

            split_total += count


            print(
                f"{emotion:10s}: {count}"
            )


        print(
            f"{'TOTAL':10s}: {split_total}"
        )


    return dataset_ok


# ============================================================
# LOAD DATASET SPLIT
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


        files = sorted(

            [

                file

                for file in os.listdir(
                    emotion_dir
                )

                if file.lower().endswith(
                    ".npy"
                )

            ]

        )


        for filename in files:

            file_path = os.path.join(
                emotion_dir,
                filename
            )


            sequence = np.load(
                file_path
            )


            # =================================================
            # EXPECTED SHAPE:
            # (20, 256)
            # =================================================

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_DIM
            ):

                raise ValueError(

                    f"Invalid sequence shape\n"
                    f"File: {file_path}\n"
                    f"Shape: {sequence.shape}\n"
                    f"Expected: "
                    f"({SEQUENCE_LENGTH}, "
                    f"{FEATURE_DIM})"

                )


            x_data.append(

                sequence.astype(
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
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(
    x_train,
    x_val,
    x_test
):

    print("")
    print("=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)


    # ========================================================
    # IMPORTANT:
    # Mean / Std فقط از TRAIN محاسبه می‌شوند.
    #
    # Validation و Test حق ندارند روی آمار
    # Normalization اثر بگذارند.
    # ========================================================

    feature_mean = np.mean(
        x_train,
        axis=(0, 1)
    )


    feature_std = np.std(
        x_train,
        axis=(0, 1)
    )


    # جلوگیری از تقسیم بر صفر

    feature_std = np.where(
        feature_std < 1e-6,
        1e-6,
        feature_std
    )


    x_train = (
        x_train
        - feature_mean
    ) / feature_std


    x_val = (
        x_val
        - feature_mean
    ) / feature_std


    x_test = (
        x_test
        - feature_mean
    ) / feature_std


    # ========================================================
    # SAVE NORMALIZATION VALUES
    # برای Webcam نهایی لازم داریم.
    # ========================================================

    np.save(
        MEAN_PATH,
        feature_mean
    )

    np.save(
        STD_PATH,
        feature_std
    )


    print(
        "[SUCCESS] Feature normalization completed."
    )

    print(
        "[SUCCESS] Mean saved:"
    )

    print(
        MEAN_PATH
    )

    print(
        "[SUCCESS] Std saved:"
    )

    print(
        STD_PATH
    )


    return (
        x_train.astype(
            np.float32
        ),

        x_val.astype(
            np.float32
        ),

        x_test.astype(
            np.float32
        )
    )


# ============================================================
# SOFT CLASS WEIGHTS
# ============================================================

def get_soft_class_weights(
    y_train
):

    print("")
    print("=" * 70)
    print("SOFT CLASS WEIGHTS")
    print("=" * 70)


    # ========================================================
    # Standard balanced weights
    # ========================================================

    raw_weights = compute_class_weight(

        class_weight="balanced",

        classes=np.arange(
            NUM_CLASSES
        ),

        y=y_train

    )


    # ========================================================
    # مهم:
    # وزن کامل قبلی باعث می‌شد Neutral بیش از حد
    # توسط مدل پیش‌بینی شود.
    #
    # sqrt وزن‌ها را ملایم می‌کند.
    # ========================================================

    soft_weights = np.sqrt(
        raw_weights
    )


    # ========================================================
    # Average weight = 1
    # ========================================================

    soft_weights = (
        soft_weights
        / np.mean(
            soft_weights
        )
    )


    class_weights = {

        index:
            float(weight)

        for index, weight
        in enumerate(
            soft_weights
        )

    }


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
        units=48,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.units = units


        self.attention_hidden = (
            tf.keras.layers.Dense(
                units,
                activation="tanh"
            )
        )


        self.attention_score = (
            tf.keras.layers.Dense(
                1
            )
        )


    def call(
        self,
        inputs
    ):

        # ====================================================
        # inputs:
        #
        # batch × time × features
        # ====================================================

        hidden = self.attention_hidden(
            inputs
        )


        scores = self.attention_score(
            hidden
        )


        # ====================================================
        # وزن هر Frame
        # ====================================================

        attention_weights = tf.nn.softmax(
            scores,
            axis=1
        )


        # ====================================================
        # Weighted sum across time
        # ====================================================

        context = tf.reduce_sum(

            inputs
            * attention_weights,

            axis=1

        )


        return context


    def get_config(
        self
    ):

        config = super().get_config()

        config.update({

            "units":
                self.units

        })

        return config


# ============================================================
# BUILD V3 MODEL
# ============================================================

def build_model():

    print("")
    print("=" * 70)
    print("BUILDING BiLSTM + ATTENTION V3")
    print("=" * 70)


    # ========================================================
    # INPUT
    # ========================================================

    inputs = tf.keras.Input(

        shape=(

            SEQUENCE_LENGTH,

            FEATURE_DIM

        ),

        name="sequence_features"

    )


    # ========================================================
    # SMALL FEATURE NOISE
    #
    # جلوگیری از حفظ کردن دقیق Featureها
    # ========================================================

    x = tf.keras.layers.GaussianNoise(
        0.01,
        name="feature_noise"
    )(inputs)


    # ========================================================
    # BiLSTM 1
    # ========================================================

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            96,

            return_sequences=True,

            dropout=0.25

        ),

        name="bilstm_1"

    )(x)


    # ========================================================
    # BiLSTM 2
    # ========================================================

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            48,

            return_sequences=True,

            dropout=0.25

        ),

        name="bilstm_2"

    )(x)


    # ========================================================
    # NORMALIZATION
    # ========================================================

    x = tf.keras.layers.LayerNormalization(

        name="temporal_normalization"

    )(x)


    # ========================================================
    # TEMPORAL ATTENTION
    # ========================================================

    context = TemporalAttention(

        units=48,

        name="temporal_attention"

    )(x)


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = tf.keras.layers.Dense(

        96,

        activation="relu",

        kernel_regularizer=(
            tf.keras.regularizers.l2(
                2e-4
            )
        ),

        name="dense_96"

    )(context)


    x = tf.keras.layers.LayerNormalization(

        name="dense_normalization"

    )(x)


    x = tf.keras.layers.Dropout(

        0.40,

        name="classifier_dropout"

    )(x)


    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = tf.keras.layers.Dense(

        NUM_CLASSES,

        activation="softmax",

        name="emotion_output"

    )(x)


    model = tf.keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="Emotion_BiLSTM_Attention_V3"

    )


    return model


# ============================================================
# VALIDATION MACRO-F1 CALLBACK
# ============================================================

class ValidationMacroF1(
    tf.keras.callbacks.Callback
):

    def __init__(
        self,
        x_val,
        y_val
    ):

        super().__init__()

        self.x_val = x_val

        self.y_val = y_val


    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        if logs is None:
            return


        probabilities = self.model.predict(

            self.x_val,

            batch_size=BATCH_SIZE,

            verbose=0

        )


        predictions = np.argmax(

            probabilities,

            axis=1

        )


        macro_f1 = f1_score(

            self.y_val,

            predictions,

            average="macro",

            zero_division=0

        )


        logs[
            "val_macro_f1"
        ] = float(
            macro_f1
        )


        print(
            ""
        )

        print(

            f"[VAL] Macro F1: "
            f"{macro_f1:.4f}"

        )


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(
    x_val,
    y_val
):

    macro_callback = ValidationMacroF1(

        x_val,

        y_val

    )


    checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(

            BEST_MODEL_PATH,

            monitor="val_macro_f1",

            mode="max",

            save_best_only=True,

            verbose=1

        )
    )


    early_stopping = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_macro_f1",

            mode="max",

            patience=8,

            restore_best_weights=True,

            verbose=1

        )
    )


    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            mode="min",

            factor=0.5,

            patience=3,

            min_lr=1e-6,

            verbose=1

        )
    )


    csv_logger = (
        tf.keras.callbacks.CSVLogger(

            os.path.join(

                RESULT_DIR,

                "lstm_attention_v3_training.csv"

            )

        )
    )


    # IMPORTANT:
    # Macro-F1 callback must run BEFORE checkpoint.
    return [

        macro_callback,

        checkpoint,

        early_stopping,

        reduce_lr,

        csv_logger

    ]


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

def save_training_graphs(
    history
):

    # ========================================================
    # ACCURACY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        history.history[
            "accuracy"
        ],

        label="Train Accuracy"

    )


    plt.plot(

        history.history[
            "val_accuracy"
        ],

        label="Validation Accuracy"

    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "LSTM Attention V3 Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "lstm_attention_v3_accuracy.png"

        ),

        dpi=200

    )


    plt.close()


    # ========================================================
    # LOSS
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        history.history[
            "loss"
        ],

        label="Train Loss"

    )


    plt.plot(

        history.history[
            "val_loss"
        ],

        label="Validation Loss"

    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "LSTM Attention V3 Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "lstm_attention_v3_loss.png"

        ),

        dpi=200

    )


    plt.close()


    # ========================================================
    # VALIDATION MACRO F1
    # ========================================================

    if (
        "val_macro_f1"
        in history.history
    ):

        plt.figure(
            figsize=(10, 6)
        )


        plt.plot(

            history.history[
                "val_macro_f1"
            ],

            label="Validation Macro F1"

        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Macro F1"
        )

        plt.title(
            "LSTM Attention V3 Validation Macro F1"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()


        plt.savefig(

            os.path.join(

                RESULT_DIR,

                "lstm_attention_v3_macro_f1.png"

            ),

            dpi=200

        )


        plt.close()


# ============================================================
# FINAL TEST
# ============================================================

def evaluate_test(
    model,
    x_test,
    y_test
):

    print("")
    print("=" * 70)

    print(
        "FINAL TEST - V3 - UNSEEN ACTORS"
    )

    print("=" * 70)


    probabilities = model.predict(

        x_test,

        batch_size=BATCH_SIZE,

        verbose=0

    )


    predictions = np.argmax(

        probabilities,

        axis=1

    )


    accuracy = accuracy_score(

        y_test,

        predictions

    )


    macro_f1 = f1_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0

    )


    weighted_f1 = f1_score(

        y_test,

        predictions,

        average="weighted",

        zero_division=0

    )


    macro_precision = precision_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0

    )


    macro_recall = recall_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0

    )


    print("")
    print(
        f"Test Accuracy     : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}"
    )


    report = classification_report(

        y_test,

        predictions,

        target_names=EMOTIONS,

        zero_division=0

    )


    print("")
    print(
        report
    )


    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(

        REPORT_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            f"Accuracy: "
            f"{accuracy:.6f}\n"

        )

        file.write(

            f"Macro Precision: "
            f"{macro_precision:.6f}\n"

        )

        file.write(

            f"Macro Recall: "
            f"{macro_recall:.6f}\n"

        )

        file.write(

            f"Macro F1: "
            f"{macro_f1:.6f}\n"

        )

        file.write(

            f"Weighted F1: "
            f"{weighted_f1:.6f}\n\n"

        )

        file.write(
            report
        )


    # ========================================================
    # SAVE JSON METRICS
    # ========================================================

    metrics = {

        "accuracy":
            float(accuracy),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1)

    }


    with open(

        METRICS_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4

        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(

        y_test,

        predictions

    )


    plt.figure(
        figsize=(9, 8)
    )


    plt.imshow(
        matrix
    )


    plt.title(
        "LSTM Attention V3 Confusion Matrix"
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

        for column in range(
            matrix.shape[1]
        ):

            plt.text(

                column,

                row,

                str(
                    matrix[
                        row,
                        column
                    ]
                ),

                ha="center",

                va="center"

            )


    plt.tight_layout()


    plt.savefig(

        CONFUSION_MATRIX_PATH,

        dpi=200

    )


    plt.close()


    return (
        accuracy,
        macro_f1
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def train():

    print("")
    print("=" * 70)

    print(
        "BiLSTM + TEMPORAL ATTENTION V3"
    )

    print("=" * 70)


    # ========================================================
    # DATA CHECK
    # ========================================================

    if not check_dataset():

        print(
            "[STOP] Dataset check failed."
        )

        return


    # ========================================================
    # LOAD DATA
    # ========================================================

    print("")
    print(
        "[INFO] Loading TRAIN..."
    )

    x_train, y_train = load_split(
        TRAIN_DIR
    )


    print(
        "[INFO] Loading VALIDATION..."
    )

    x_val, y_val = load_split(
        VAL_DIR
    )


    print(
        "[INFO] Loading TEST..."
    )

    x_test, y_test = load_split(
        TEST_DIR
    )


    # ========================================================
    # SHAPES BEFORE NORMALIZATION
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "DATA SHAPES"
    )

    print("=" * 70)


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


    # ========================================================
    # EXPECTED SHAPES
    # ========================================================

    if x_train.shape[1:] != (
        SEQUENCE_LENGTH,
        FEATURE_DIM
    ):

        raise RuntimeError(
            "Train feature shape is invalid."
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    (
        x_train,
        x_val,
        x_test

    ) = normalize_features(

        x_train,
        x_val,
        x_test

    )


    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    class_weights = (
        get_soft_class_weights(
            y_train
        )
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

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


    print("")
    print("=" * 70)

    print(
        "V3 MODEL SUMMARY"
    )

    print("=" * 70)


    model.summary()


    # ========================================================
    # TRAIN
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "V3 TRAINING STARTED"
    )

    print("=" * 70)


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

        callbacks=get_callbacks(
            x_val,
            y_val
        ),

        verbose=1

    )


    # ========================================================
    # Because EarlyStopping uses restore_best_weights=True,
    # current model contains the best validation weights.
    # ========================================================

    model.save(
        FINAL_MODEL_PATH
    )


    # ========================================================
    # SAVE CLASSES
    # ========================================================

    with open(

        CLASS_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            EMOTIONS,

            file,

            indent=4

        )


    # ========================================================
    # TRAINING PLOTS
    # ========================================================

    save_training_graphs(
        history
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    accuracy, macro_f1 = evaluate_test(

        model,

        x_test,

        y_test

    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "V3 TRAINING COMPLETED"
    )

    print("=" * 70)


    print("")
    print(
        "Best V3 model:"
    )

    print(
        BEST_MODEL_PATH
    )


    print("")
    print(
        "Final V3 model:"
    )

    print(
        FINAL_MODEL_PATH
    )


    print("")
    print(
        "Final Test Accuracy:"
    )

    print(
        f"{accuracy:.4f}"
    )


    print(
        "Final Test Macro F1:"
    )

    print(
        f"{macro_f1:.4f}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train()