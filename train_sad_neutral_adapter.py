import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "webcam_features"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "adapter"
)

MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_mean.npy"
)

STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_std.npy"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sad_neutral_adapter.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "sad_neutral_adapter_classes.json"
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "sad_neutral_adapter_metrics.json"
)

REPORT_PATH = os.path.join(
    RESULT_DIR,
    "sad_neutral_adapter_cv_report.txt"
)

# فقط این دو کلاس وارد آموزش می‌شوند
CLASSES = [
    "neutral",
    "sad"
]

SEQUENCE_LENGTH = 20
FEATURE_DIM = 256

BATCH_SIZE = 8
MAX_EPOCHS = 40
LEARNING_RATE = 2e-4


# ============================================================
# UTILITIES
# ============================================================

def ensure_exists(path, label):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def get_persons():

    ensure_exists(
        DATA_ROOT,
        "Webcam features folder"
    )

    persons = []

    for name in sorted(
        os.listdir(DATA_ROOT)
    ):

        full_path = os.path.join(
            DATA_ROOT,
            name
        )

        if os.path.isdir(full_path):
            persons.append(name)

    return persons


def load_person(person_name):

    x_data = []
    y_data = []

    person_dir = os.path.join(
        DATA_ROOT,
        person_name
    )

    for class_index, emotion in enumerate(
        CLASSES
    ):

        emotion_dir = os.path.join(
            person_dir,
            emotion
        )

        if not os.path.isdir(
            emotion_dir
        ):

            raise FileNotFoundError(
                f"Missing emotion folder:\n{emotion_dir}"
            )

        files = sorted([
            filename
            for filename in os.listdir(
                emotion_dir
            )
            if filename.lower().endswith(
                ".npy"
            )
        ])

        if len(files) == 0:

            raise RuntimeError(
                f"No .npy files found in:\n{emotion_dir}"
            )

        for filename in files:

            path = os.path.join(
                emotion_dir,
                filename
            )

            sequence = np.load(
                path
            ).astype(
                np.float32
            )

            expected_shape = (
                SEQUENCE_LENGTH,
                FEATURE_DIM
            )

            if sequence.shape != expected_shape:

                raise ValueError(
                    f"Invalid shape in {path}: "
                    f"{sequence.shape}, "
                    f"expected {expected_shape}"
                )

            x_data.append(
                sequence
            )

            y_data.append(
                class_index
            )

    return (
        np.asarray(
            x_data,
            dtype=np.float32
        ),
        np.asarray(
            y_data,
            dtype=np.int32
        )
    )


def normalize(
    x,
    feature_mean,
    feature_std
):

    return (
        (
            x
            - feature_mean
        )
        / feature_std
    ).astype(
        np.float32
    )


# ============================================================
# BINARY ADAPTER MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_DIM
        ),
        name="sad_neutral_sequence"
    )

    # نویز خیلی کم برای جلوگیری از حفظ کردن دقیق نمونه‌ها
    x = tf.keras.layers.GaussianNoise(
        0.008,
        name="feature_noise"
    )(inputs)

    # مدل کوچک نگه داشته شده چون فقط دو نفر داده داریم
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            24,
            return_sequences=True,
            dropout=0.20
        ),
        name="sad_neutral_bigru"
    )(x)

    x = tf.keras.layers.LayerNormalization(
        name="sad_neutral_norm"
    )(x)

    avg_pool = (
        tf.keras.layers.GlobalAveragePooling1D(
            name="avg_pool"
        )(x)
    )

    max_pool = (
        tf.keras.layers.GlobalMaxPooling1D(
            name="max_pool"
        )(x)
    )

    x = tf.keras.layers.Concatenate(
        name="pool_fusion"
    )([
        avg_pool,
        max_pool
    ])

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=(
            tf.keras.regularizers.l2(
                2e-4
            )
        ),
        name="adapter_dense"
    )(x)

    x = tf.keras.layers.Dropout(
        0.35,
        name="adapter_dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="sad_neutral_output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="SadNeutralAdapter"
    )

    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            )
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    print("")
    print("=" * 72)
    print("SAD / NEUTRAL ADAPTER")
    print("ONLY THESE TWO CLASSES ARE TRAINED")
    print("=" * 72)

    ensure_exists(
        MEAN_PATH,
        "V1 feature mean"
    )

    ensure_exists(
        STD_PATH,
        "V1 feature std"
    )

    feature_mean = np.load(
        MEAN_PATH
    ).astype(
        np.float32
    )

    feature_std = np.load(
        STD_PATH
    ).astype(
        np.float32
    )

    feature_std = np.where(
        feature_std < 1e-6,
        1e-6,
        feature_std
    )

    persons = get_persons()

    if len(persons) < 2:

        raise RuntimeError(
            "At least 2 person folders are required "
            "for person-wise validation."
        )

    print("")
    print(
        "Persons found:",
        persons
    )

    print("")
    print(
        "IMPORTANT:"
    )

    print(
        "Fear is NOT used here."
    )

    print(
        "Angry / Disgust / Happy / Surprise are NOT used here."
    )

    person_data = {}

    for person in persons:

        x_person, y_person = (
            load_person(
                person
            )
        )

        x_person = normalize(
            x_person,
            feature_mean,
            feature_std
        )

        person_data[
            person
        ] = (
            x_person,
            y_person
        )

        neutral_count = int(
            np.sum(
                y_person == 0
            )
        )

        sad_count = int(
            np.sum(
                y_person == 1
            )
        )

        print(
            f"{person}: "
            f"{len(y_person)} sequences "
            f"(neutral={neutral_count}, sad={sad_count})"
        )

    # ========================================================
    # PERSON-WISE CROSS VALIDATION
    # ========================================================

    print("")
    print("=" * 72)
    print("PERSON-WISE CROSS VALIDATION")
    print("=" * 72)

    fold_results = []
    best_epochs = []
    report_lines = []

    for fold_index, val_person in enumerate(
        persons,
        start=1
    ):

        train_persons = [
            person
            for person in persons
            if person != val_person
        ]

        x_train = np.concatenate(
            [
                person_data[person][0]
                for person in train_persons
            ],
            axis=0
        )

        y_train = np.concatenate(
            [
                person_data[person][1]
                for person in train_persons
            ],
            axis=0
        )

        x_val, y_val = (
            person_data[
                val_person
            ]
        )

        print("")
        print(
            f"FOLD {fold_index}/{len(persons)}"
        )

        print(
            "Train persons:",
            train_persons
        )

        print(
            "Validation person:",
            val_person
        )

        print(
            "Train sequences:",
            len(y_train)
        )

        print(
            "Validation sequences:",
            len(y_val)
        )

        tf.keras.backend.clear_session()

        model = build_model()

        early_stopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=7,
                restore_best_weights=True,
                verbose=0
            )
        )

        reduce_lr = (
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0
            )
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(
                x_val,
                y_val
            ),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[
                early_stopping,
                reduce_lr
            ],
            verbose=0
        )

        best_epoch = (
            int(
                np.argmin(
                    history.history[
                        "val_loss"
                    ]
                )
            )
            + 1
        )

        best_epochs.append(
            best_epoch
        )

        probabilities = model.predict(
            x_val,
            batch_size=BATCH_SIZE,
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

        per_class_f1 = f1_score(
            y_val,
            predictions,
            average=None,
            labels=[
                0,
                1
            ],
            zero_division=0
        )

        matrix = confusion_matrix(
            y_val,
            predictions,
            labels=[
                0,
                1
            ]
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Macro F1 : {macro_f1:.4f}"
        )

        print(
            f"neutral  : F1 {per_class_f1[0]:.4f}"
        )

        print(
            f"sad      : F1 {per_class_f1[1]:.4f}"
        )

        print(
            "Best epoch:",
            best_epoch
        )

        fold_results.append({
            "validation_person":
                val_person,

            "train_persons":
                train_persons,

            "accuracy":
                float(
                    accuracy
                ),

            "macro_f1":
                float(
                    macro_f1
                ),

            "neutral_f1":
                float(
                    per_class_f1[0]
                ),

            "sad_f1":
                float(
                    per_class_f1[1]
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "confusion_matrix":
                matrix.tolist()
        })

        report = classification_report(
            y_val,
            predictions,
            target_names=CLASSES,
            zero_division=0
        )

        report_lines.append(
            "\n"
            + "=" * 72
            + "\n"
            + f"VALIDATION PERSON: {val_person}\n"
            + "=" * 72
            + "\n"
            + report
            + "\n"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    mean_accuracy = float(
        np.mean([
            result[
                "accuracy"
            ]
            for result in fold_results
        ])
    )

    mean_macro_f1 = float(
        np.mean([
            result[
                "macro_f1"
            ]
            for result in fold_results
        ])
    )

    mean_neutral_f1 = float(
        np.mean([
            result[
                "neutral_f1"
            ]
            for result in fold_results
        ])
    )

    mean_sad_f1 = float(
        np.mean([
            result[
                "sad_f1"
            ]
            for result in fold_results
        ])
    )

    final_epochs = max(
        3,
        int(
            round(
                np.mean(
                    best_epochs
                )
            )
        )
    )

    print("")
    print("=" * 72)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 72)

    print(
        f"Mean Accuracy   : "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Mean Macro F1   : "
        f"{mean_macro_f1:.4f}"
    )

    print(
        f"Mean Neutral F1 : "
        f"{mean_neutral_f1:.4f}"
    )

    print(
        f"Mean Sad F1     : "
        f"{mean_sad_f1:.4f}"
    )

    print(
        "Final training epochs:",
        final_epochs
    )

    # ========================================================
    # FINAL TRAINING ON ALL SAD + NEUTRAL DATA
    # ========================================================

    x_all = np.concatenate(
        [
            person_data[person][0]
            for person in persons
        ],
        axis=0
    )

    y_all = np.concatenate(
        [
            person_data[person][1]
            for person in persons
        ],
        axis=0
    )

    tf.keras.backend.clear_session()

    final_model = build_model()

    print("")
    print("=" * 72)
    print("TRAINING FINAL SAD / NEUTRAL ADAPTER")
    print("=" * 72)

    print(
        "All training sequences:",
        len(y_all)
    )

    final_model.fit(
        x_all,
        y_all,
        epochs=final_epochs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1
    )

    final_model.save(
        FINAL_MODEL_PATH
    )

    with open(
        CLASS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            CLASSES,
            file,
            indent=4
        )

    metrics = {
        "classes":
            CLASSES,

        "persons":
            persons,

        "folds":
            fold_results,

        "mean_accuracy":
            mean_accuracy,

        "mean_macro_f1":
            mean_macro_f1,

        "mean_neutral_f1":
            mean_neutral_f1,

        "mean_sad_f1":
            mean_sad_f1,

        "final_training_epochs":
            final_epochs,

        "important_note":
            (
                "This adapter is trained only on "
                "neutral and sad webcam features. "
                "Fear is excluded. Strong V1 model "
                "weights are not modified."
            )
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

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

        file.write(
            "\n"
            + "=" * 72
            + "\n"
        )

        file.write(
            f"Mean Accuracy: "
            f"{mean_accuracy:.6f}\n"
        )

        file.write(
            f"Mean Macro F1: "
            f"{mean_macro_f1:.6f}\n"
        )

        file.write(
            f"Mean Neutral F1: "
            f"{mean_neutral_f1:.6f}\n"
        )

        file.write(
            f"Mean Sad F1: "
            f"{mean_sad_f1:.6f}\n"
        )

        file.write(
            f"Final epochs: "
            f"{final_epochs}\n"
        )

    print("")
    print("=" * 72)
    print("SAD / NEUTRAL ADAPTER READY")
    print("=" * 72)

    print(
        "Model saved:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print("")
    print(
        "Classes:"
    )

    print(
        CLASSES
    )

    print("")
    print(
        "Fear was excluded from this adapter."
    )

    print(
        "Original V1 model was NOT modified."
    )


if __name__ == "__main__":
    main()
