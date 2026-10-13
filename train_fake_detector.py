import os
import shutil
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from fake_detector import (
    build_discriminator,
    IMAGE_SIZE,
    DISCRIMINATOR_MODEL_PATH
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 40

VALIDATION_SPLIT = 0.20

DATA_FOLDER = "data"

CAPTURED_FACE_FOLDER = os.path.join(
    DATA_FOLDER,
    "captured_faces"
)

DATASET_FOLDER = os.path.join(
    DATA_FOLDER,
    "fake_dataset"
)

FAKE_FOLDER = os.path.join(
    DATASET_FOLDER,
    "fake"
)

REAL_FOLDER = os.path.join(
    DATASET_FOLDER,
    "real"
)

MODEL_FOLDER = "models"

RESULTS_FOLDER = os.path.join("results", "anti_spoofing")

BEST_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fake_discriminator.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fake_discriminator_final.keras"
)

HISTORY_IMAGE_PATH = os.path.join(
    RESULTS_FOLDER,
    "fake_detector_training.png"
)

METRICS_PATH = os.path.join(
    RESULTS_FOLDER,
    "fake_detector_metrics.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_FOLDER,
    "fake_detector_confusion_matrix.png"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    folders = [
        DATASET_FOLDER,
        REAL_FOLDER,
        FAKE_FOLDER,
        MODEL_FOLDER,
        RESULTS_FOLDER
    ]

    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )


# ============================================================
# CHECK IMAGE EXTENSION
# ============================================================

def is_image(filename):

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    return filename.lower().endswith(
        valid_extensions
    )


# ============================================================
# COPY CAPTURED REAL FACES
# ============================================================

def copy_captured_faces():

    if not os.path.exists(
        CAPTURED_FACE_FOLDER
    ):

        print(
            "[WARNING] captured_faces folder not found."
        )

        return


    source_images = [

        file

        for file in os.listdir(
            CAPTURED_FACE_FOLDER
        )

        if is_image(file)
    ]


    copied = 0


    for filename in source_images:

        source_path = os.path.join(
            CAPTURED_FACE_FOLDER,
            filename
        )

        destination_path = os.path.join(
            REAL_FOLDER,
            filename
        )


        if not os.path.exists(
            destination_path
        ):

            shutil.copy2(
                source_path,
                destination_path
            )

            copied += 1


    print(
        f"[INFO] {copied} captured real faces copied."
    )


# ============================================================
# COUNT DATASET
# ============================================================

def count_images(folder):

    if not os.path.exists(folder):

        return 0


    return len(

        [

            file

            for file in os.listdir(folder)

            if is_image(file)

        ]

    )


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    real_count = count_images(
        REAL_FOLDER
    )

    fake_count = count_images(
        FAKE_FOLDER
    )


    print("")
    print("=" * 60)

    print("DATASET INFORMATION")

    print("=" * 60)

    print(
        f"Real images : {real_count}"
    )

    print(
        f"Fake images : {fake_count}"
    )


    if real_count < 20:

        print("")
        print(
            "[ERROR] Not enough REAL images."
        )

        print(
            "Run main.py and collect more real faces."
        )

        return False


    if fake_count < 20:

        print("")
        print(
            "[ERROR] Not enough FAKE images."
        )

        print("")
        print(
            "Put fake/spoof face images inside:"
        )

        print(
            FAKE_FOLDER
        )

        print("")
        print(
            "Examples:"
        )

        print(
            "- Face displayed on a phone"
        )

        print(
            "- Face displayed on a monitor"
        )

        print(
            "- Printed face photo"
        )

        print(
            "- AI generated face"
        )

        print(
            "- Deepfake sample"
        )

        return False


    return True


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.08
        ),

        tf.keras.layers.RandomContrast(
            0.15
        )

    ],

    name="Data_Augmentation"

)


# ============================================================
# PREPROCESS TRAIN DATA
# ============================================================

def preprocess_train(
    images,
    labels
):

    images = tf.cast(
        images,
        tf.float32
    )


    images = data_augmentation(
        images,
        training=True
    )


    # --------------------------------------------------------
    # Normalize exactly like fake_detector.py
    # Range = -1 to +1
    # --------------------------------------------------------

    images = (
        images - 127.5
    ) / 127.5


    return images, labels


# ============================================================
# PREPROCESS VALIDATION DATA
# ============================================================

def preprocess_validation(
    images,
    labels
):

    images = tf.cast(
        images,
        tf.float32
    )


    images = (
        images - 127.5
    ) / 127.5


    return images, labels


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    print("")
    print(
        "[INFO] Loading dataset..."
    )


    # IMPORTANT:
    # fake = 0
    # real = 1

    class_names = [
        "fake",
        "real"
    ]


    train_dataset = (
        tf.keras.utils.image_dataset_from_directory(

            DATASET_FOLDER,

            labels="inferred",

            label_mode="binary",

            class_names=class_names,

            validation_split=VALIDATION_SPLIT,

            subset="training",

            seed=SEED,

            image_size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            batch_size=BATCH_SIZE,

            shuffle=True

        )
    )


    validation_dataset = (
        tf.keras.utils.image_dataset_from_directory(

            DATASET_FOLDER,

            labels="inferred",

            label_mode="binary",

            class_names=class_names,

            validation_split=VALIDATION_SPLIT,

            subset="validation",

            seed=SEED,

            image_size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            batch_size=BATCH_SIZE,

            shuffle=False

        )
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    train_dataset = train_dataset.map(

        preprocess_train,

        num_parallel_calls=tf.data.AUTOTUNE

    )


    validation_dataset = validation_dataset.map(

        preprocess_validation,

        num_parallel_calls=tf.data.AUTOTUNE

    )


    # --------------------------------------------------------
    # Improve performance
    # --------------------------------------------------------

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )


    return (
        train_dataset,
        validation_dataset
    )


# ============================================================
# BUILD MODEL
# ============================================================

def create_model():

    print("")
    print(
        "[INFO] Building discriminator..."
    )


    model = build_discriminator()


    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=0.0002,

            beta_1=0.5

        ),

        loss="binary_crossentropy",

        metrics=[

            "accuracy",

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            )

        ]

    )


    return model


# ============================================================
# CALLBACKS
# ============================================================

def create_callbacks():

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=7,

            restore_best_weights=True,

            verbose=1

        )
    )


    reduce_learning_rate = (
        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            min_lr=1e-7,

            verbose=1

        )
    )


    model_checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(

            BEST_MODEL_PATH,

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        )
    )


    return [

        early_stopping,

        reduce_learning_rate,

        model_checkpoint

    ]


# ============================================================
# DRAW TRAINING HISTORY
# ============================================================

def save_training_history(
    history
):

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
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
        "Real / Fake Detector Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        HISTORY_IMAGE_PATH,
        dpi=200
    )

    plt.close()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_path = os.path.join(
        RESULTS_FOLDER,
        "fake_detector_loss.png"
    )


    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
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
        "Real / Fake Detector Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        loss_path,
        dpi=200
    )

    plt.close()


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    validation_dataset
):

    print("")
    print("=" * 60)

    print(
        "FINAL EVALUATION"
    )

    print("=" * 60)


    true_labels = []

    predicted_labels = []


    for images, labels in validation_dataset:

        predictions = model.predict(

            images,

            verbose=0

        )


        predictions = (
            predictions.reshape(-1)
            >= 0.5
        ).astype(int)


        labels = (
            labels.numpy()
            .reshape(-1)
            .astype(int)
        )


        true_labels.extend(
            labels.tolist()
        )

        predicted_labels.extend(
            predictions.tolist()
        )


    accuracy = accuracy_score(

        true_labels,
        predicted_labels

    )


    precision = precision_score(

        true_labels,
        predicted_labels,

        zero_division=0

    )


    recall = recall_score(

        true_labels,
        predicted_labels,

        zero_division=0

    )


    f1 = f1_score(

        true_labels,
        predicted_labels,

        zero_division=0

    )


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    print("")
    print(
        classification_report(

            true_labels,

            predicted_labels,

            target_names=[
                "Fake",
                "Real"
            ],

            zero_division=0

        )
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1)

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


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(

        true_labels,

        predicted_labels

    )


    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Fake Detector Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        [
            "Fake",
            "Real"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "Fake",
            "Real"
        ]
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


# ============================================================
# TRAIN
# ============================================================

def train():

    print("")
    print("=" * 60)

    print(
        "FACIAL REAL / FAKE DETECTOR TRAINING"
    )

    print("=" * 60)


    create_directories()


    # --------------------------------------------------------
    # Use previously captured faces as REAL samples
    # --------------------------------------------------------

    copy_captured_faces()


    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    dataset_ready = check_dataset()


    if not dataset_ready:

        print("")
        print(
            "[STOP] Dataset is not ready."
        )

        return


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset

    ) = create_datasets()


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()


    model.summary()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("")
    print(
        "[INFO] Training started..."
    )


    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=create_callbacks()

    )


    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(
        FINAL_MODEL_PATH
    )


    # --------------------------------------------------------
    # Save graphs
    # --------------------------------------------------------

    save_training_history(
        history
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluate_model(

        model,

        validation_dataset

    )


    print("")
    print("=" * 60)

    print(
        "TRAINING COMPLETED"
    )

    print("=" * 60)


    print("")
    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )


    print("")
    print(
        "Final model:"
    )

    print(
        FINAL_MODEL_PATH
    )


    print("")
    print(
        "Training results:"
    )

    print(
        RESULTS_FOLDER
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train()