import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "cnn_actor_split"
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
    "cnn"
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
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 160

BATCH_SIZE = 16

PHASE1_EPOCHS = 20

PHASE2_EPOCHS = 20

SEED = 42


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
# OUTPUT FILES
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_cnn_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_cnn_final.keras"
)

FEATURE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cnn_feature_extractor.keras"
)

CLASS_FILE_PATH = os.path.join(
    MODEL_DIR,
    "emotion_classes.json"
)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print("")
    print("=" * 70)
    print("CHECKING CNN ACTOR-WISE DATASET")
    print("=" * 70)

    for split_name, split_path in [

        ("Train", TRAIN_DIR),

        ("Validation", VAL_DIR),

        ("Test", TEST_DIR)

    ]:

        print("")
        print(split_name.upper())
        print("-" * 40)

        total = 0

        for emotion in EMOTIONS:

            folder = os.path.join(
                split_path,
                emotion
            )

            if not os.path.isdir(folder):

                print(
                    "[ERROR] Missing:",
                    folder
                )

                return False

            count = len(
                [
                    f
                    for f in os.listdir(folder)

                    if f.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".bmp",
                            ".webp"
                        )
                    )
                ]
            )

            total += count

            print(
                f"{emotion:10s}: {count}"
            )

        print(
            f"{'TOTAL':10s}: {total}"
        )

    return True


# ============================================================
# DATASETS
# ============================================================

def load_datasets():

    print("")
    print(
        "[INFO] Loading datasets..."
    )

    train_ds = (
        tf.keras.utils.image_dataset_from_directory(

            TRAIN_DIR,

            class_names=EMOTIONS,

            labels="inferred",

            label_mode="int",

            color_mode="rgb",

            image_size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            batch_size=BATCH_SIZE,

            shuffle=True,

            seed=SEED

        )
    )


    val_ds = (
        tf.keras.utils.image_dataset_from_directory(

            VAL_DIR,

            class_names=EMOTIONS,

            labels="inferred",

            label_mode="int",

            color_mode="rgb",

            image_size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            batch_size=BATCH_SIZE,

            shuffle=False

        )
    )


    test_ds = (
        tf.keras.utils.image_dataset_from_directory(

            TEST_DIR,

            class_names=EMOTIONS,

            labels="inferred",

            label_mode="int",

            color_mode="rgb",

            image_size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            batch_size=BATCH_SIZE,

            shuffle=False

        )
    )


    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        tf.data.AUTOTUNE
    )


    return (
        train_ds,
        val_ds,
        test_ds
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights():

    labels = []


    for index, emotion in enumerate(
        EMOTIONS
    ):

        folder = os.path.join(
            TRAIN_DIR,
            emotion
        )

        count = len(
            [
                f
                for f in os.listdir(folder)

                if f.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp"
                    )
                )
            ]
        )

        labels.extend(
            [index] * count
        )


    labels = np.array(
        labels
    )


    weights = compute_class_weight(

        class_weight="balanced",

        classes=np.arange(
            NUM_CLASSES
        ),

        y=labels

    )


    class_weights = {

        i: float(weight)

        for i, weight
        in enumerate(weights)

    }


    print("")
    print("CLASS WEIGHTS")
    print("-" * 40)


    for index, weight in class_weights.items():

        print(
            f"{EMOTIONS[index]:10s}: "
            f"{weight:.3f}"
        )


    return class_weights


# ============================================================
# AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.04
        ),

        tf.keras.layers.RandomZoom(
            0.08
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        tf.keras.layers.RandomContrast(
            0.12
        )

    ],

    name="emotion_augmentation"

)


# ============================================================
# MODEL
# ============================================================

def build_model():

    print("")
    print("[INFO] Building MobileNetV2 emotion CNN...")

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="rgb_face"
    )

    x = augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        x
    )

    weights_path = os.path.join(
        MODEL_DIR,
        "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    )

    if os.path.isfile(weights_path):

        print("[INFO] Using local MobileNetV2 ImageNet weights.")

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights=None
        )

        base_model.load_weights(weights_path)

    else:

        print(
            "[INFO] Local weights not found; "
            "using Keras ImageNet weights."
        )

        try:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                include_top=False,
                weights="imagenet"
            )
        except Exception as error:
            raise RuntimeError(
                "Could not load MobileNetV2 ImageNet weights. "
                "Either allow Keras to download them or place the "
                "160x160 no-top MobileNetV2 weights file in models/. "
                f"Original error: {error}"
            ) from error

    base_model.trainable = False

    x = base_model(
        x,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_features"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    features = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="feature_embedding"
    )(x)

    features = tf.keras.layers.BatchNormalization()(
        features
    )

    features = tf.keras.layers.Dropout(
        0.40
    )(features)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="emotion_output"
    )(features)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Emotion_MobileNetV2"
    )

    return model, base_model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(
    phase_name
):

    checkpoint_path = os.path.join(

        MODEL_DIR,

        f"{phase_name}_best.keras"

    )


    return [

        tf.keras.callbacks.ModelCheckpoint(

            checkpoint_path,

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            verbose=1

        ),


        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True,

            verbose=1

        ),


        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.35,

            patience=2,

            min_lr=1e-7,

            verbose=1

        )

    ]


# ============================================================
# TRAINING GRAPH
# ============================================================

def save_history(
    history,
    name
):

    # Accuracy

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
        name + " Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            RESULT_DIR,

            name + "_accuracy.png"

        ),

        dpi=200

    )

    plt.close()


    # Loss

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
        name + " Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            RESULT_DIR,

            name + "_loss.png"

        ),

        dpi=200

    )

    plt.close()


# ============================================================
# FINE TUNE
# ============================================================

def enable_fine_tuning(
    base_model
):

    print("")
    print(
        "[INFO] Enabling fine tuning..."
    )


    base_model.trainable = True


    # Freeze everything except last 35 layers

    for layer in base_model.layers[
        :-60
    ]:

        layer.trainable = False


    # Keep BatchNorm frozen.
    # Important for small datasets.

    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False


    trainable_count = sum(

        int(layer.trainable)

        for layer in base_model.layers

    )


    print(
        "[INFO] Trainable MobileNet layers:",
        trainable_count
    )


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(
    model,
    test_ds
):

    print("")
    print("=" * 70)

    print(
        "FINAL TEST - UNSEEN ACTORS"
    )

    print("=" * 70)


    true_labels = []

    predicted_labels = []


    for images, labels in test_ds:

        probabilities = model.predict(

            images,

            verbose=0

        )


        predictions = np.argmax(

            probabilities,

            axis=1

        )


        true_labels.extend(

            labels.numpy().tolist()

        )


        predicted_labels.extend(

            predictions.tolist()

        )


    accuracy = accuracy_score(

        true_labels,

        predicted_labels

    )


    macro_f1 = f1_score(

        true_labels,

        predicted_labels,

        average="macro"

    )


    weighted_f1 = f1_score(

        true_labels,

        predicted_labels,

        average="weighted"

    )


    print("")
    print(
        f"Test Accuracy   : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1     : "
        f"{weighted_f1:.4f}"
    )

    print("")


    report = classification_report(

        true_labels,

        predicted_labels,

        target_names=EMOTIONS,

        zero_division=0

    )


    print(
        report
    )


    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(

        os.path.join(

            RESULT_DIR,

            "cnn_test_report.txt"

        ),

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            f"Accuracy: "
            f"{accuracy:.6f}\n"

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


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(

        true_labels,

        predicted_labels

    )


    plt.figure(
        figsize=(9, 8)
    )


    plt.imshow(
        matrix
    )


    plt.title(
        "CNN Test Confusion Matrix"
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

        os.path.join(

            RESULT_DIR,

            "cnn_test_confusion_matrix.png"

        ),

        dpi=200

    )


    plt.close()


    return (
        accuracy,
        macro_f1
    )


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

def save_feature_extractor(
    model
):

    feature_layer = model.get_layer(
        "feature_embedding"
    )


    feature_model = tf.keras.Model(

        inputs=model.input,

        outputs=feature_layer.output,

        name="CNN_Feature_Extractor"

    )


    feature_model.save(
        FEATURE_MODEL_PATH
    )


    print("")
    print(
        "[SUCCESS] Feature extractor saved:"
    )

    print(
        FEATURE_MODEL_PATH
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def train():

    print("")
    print("=" * 70)

    print(
        "ACTOR-WISE FACIAL EMOTION CNN TRAINING"
    )

    print("=" * 70)


    if not check_dataset():

        return


    (
        train_ds,
        val_ds,
        test_ds

    ) = load_datasets()


    class_weights = (
        calculate_class_weights()
    )


    model, base_model = (
        build_model()
    )


    # ========================================================
    # PHASE 1
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "PHASE 1 - TRANSFER LEARNING"
    )

    print("=" * 70)


    model.compile(

        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=5e-4
            )
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]

    )


    history1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=PHASE1_EPOCHS,

        class_weight=class_weights,

        callbacks=get_callbacks(
            "cnn_phase1"
        )

    )


    save_history(

        history1,

        "cnn_phase1"

    )


    # ========================================================
    # PHASE 2
    # ========================================================

    enable_fine_tuning(
        base_model
    )


    print("")
    print("=" * 70)

    print(
        "PHASE 2 - FINE TUNING"
    )

    print("=" * 70)


    model.compile(

        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=5e-6
            )
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]

    )


    history2 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=PHASE2_EPOCHS,

        class_weight=class_weights,

        callbacks=get_callbacks(
            "cnn_phase2"
        )

    )


    save_history(

        history2,

        "cnn_phase2"

    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model.save(
        FINAL_MODEL_PATH
    )


    # Copy final as standard best path

    model.save(
        BEST_MODEL_PATH
    )


    # ========================================================
    # CLASS NAMES
    # ========================================================

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


    # ========================================================
    # FEATURE EXTRACTOR
    # ========================================================

    save_feature_extractor(
        model
    )


    # ========================================================
    # TEST
    # ========================================================

    evaluate_test(

        model,

        test_ds

    )


    print("")
    print("=" * 70)

    print(
        "CNN TRAINING COMPLETED"
    )

    print("=" * 70)


    print("")
    print(
        "Final model:"
    )

    print(
        FINAL_MODEL_PATH
    )


    print("")
    print(
        "Feature extractor:"
    )

    print(
        FEATURE_MODEL_PATH
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train()