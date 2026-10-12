import os
import json
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
# CONFIG
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
    "results"
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

SEQUENCE_LENGTH = 20

RAW_FEATURE_DIM = 256
MODEL_FEATURE_DIM = 512

# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 60
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_final.keras"
)

CLASS_FILE_PATH = os.path.join(
    MODEL_DIR,
    "lstm_emotion_classes.json"
)

NORM_MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_feature_mean.npy"
)

NORM_STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_feature_std.npy"
)


# ============================================================
# ATTENTION LAYER
# ============================================================

class TemporalAttention(
    tf.keras.layers.Layer
):
    def __init__(
        self,
        units=64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(
            units,
            activation="tanh"
        )
        self.weight_dense = tf.keras.layers.Dense(
            1
        )

    def call(
        self,
        inputs
    ):
        # inputs shape:
        # (batch, time, features)

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print("")
    print("=" * 70)
    print("CHECKING LSTM FEATURE DATASET")
    print("=" * 70)

    ok = True

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

    x = []
    y = []

    for label_index, emotion in enumerate(
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
            if f.lower().endswith(".npy")
        ])

        for filename in files:

            path = os.path.join(
                emotion_dir,
                filename
            )

            features = np.load(
                path
            )

            # Expected shape:
            # (20, 256)

            if features.shape != (
    SEQUENCE_LENGTH,
    RAW_FEATURE_DIM
):
                raise ValueError(
                    f"Invalid shape for {path}: "
                    f"{features.shape}"
                )

            x.append(
                features.astype(
                    np.float32
                )
            )

            y.append(
                label_index
            )

    x = np.array(
        x,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    return x, y


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(
    x_train,
    x_val,
    x_test
):

    # mean/std over all train samples and all time steps
    # per feature dimension

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
        NORM_MEAN_PATH,
        feature_mean
    )

    np.save(
        NORM_STD_PATH,
        feature_std
    )

    print("")
    print(
        "[SUCCESS] LSTM normalization statistics saved."
    )

    print(
        NORM_MEAN_PATH
    )

    print(
        NORM_STD_PATH
    )

    return (
        x_train,
        x_val,
        x_test
    )
def add_temporal_deltas(x):

    first_frame = x[:, 0:1, :]

    shifted = np.concatenate(
        [
            first_frame,
            x[:, :-1, :]
        ],
        axis=1
    )

    delta = x - shifted

    x = np.concatenate(
        [
            x,
            delta
        ],
        axis=-1
    )

    return x.astype(
        np.float32
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_class_weights(
    y_train
):

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train
    )

    class_weights = {
        i: float(w)
        for i, w in enumerate(weights)
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
# TF DATASET
# ============================================================

def make_tf_dataset(
    x,
    y,
    training=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (x, y)
    )

    if training:
        ds = ds.shuffle(
            buffer_size=len(x),
            seed=SEED
        )

    ds = ds.batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    )

    return ds


# ============================================================
# MODEL
# ============================================================

def build_model():
    inputs = tf.keras.Input(
    shape=(
        SEQUENCE_LENGTH,
        MODEL_FEATURE_DIM
    ),
    name="temporal_features"
)

    x = tf.keras.layers.GaussianNoise(
        0.015
    )(inputs)

    # ========================================================
    # BiLSTM
    # ========================================================

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.30
        ),
        name="bilstm_1"
    )(x)

    x = tf.keras.layers.LayerNormalization()(
        x
    )

    # ========================================================
    # MULTI-HEAD TEMPORAL ATTENTION
    # ========================================================

    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        dropout=0.20,
        name="temporal_self_attention"
    )(
        x,
        x
    )

    # Residual connection
    x = tf.keras.layers.Add()(
        [
            x,
            attention
        ]
    )

    x = tf.keras.layers.LayerNormalization()(
        x
    )

    # ========================================================
    # SECOND BiLSTM
    # ========================================================

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.30
        ),
        name="bilstm_2"
    )(x)

    # ========================================================
    # TEMPORAL POOLING
    # ========================================================

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(
        x
    )

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(
        x
    )

    x = tf.keras.layers.Concatenate()(
        [
            avg_pool,
            max_pool
        ]
    )

    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(
            2e-4
        )
    )(x)

    x = tf.keras.layers.BatchNormalization()(
        x
    )

    x = tf.keras.layers.Dropout(
        0.45
    )(x)

    outputs = tf.keras.layers.Dense(
        7,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Emotion_BiLSTM_Attention_V2"
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
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]


# ============================================================
# PLOTS
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
        "LSTM + Attention Accuracy"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "lstm_attention_accuracy.png"
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
        "LSTM + Attention Loss"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "lstm_attention_loss.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_ds,
    y_test
):

    print("")
    print("=" * 70)
    print("FINAL TEST - LSTM + ATTENTION - UNSEEN ACTORS")
    print("=" * 70)

    probabilities = model.predict(
        test_ds,
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
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
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

    print("")

    report = classification_report(
        y_test,
        y_pred,
        target_names=EMOTIONS,
        zero_division=0
    )

    print(
        report
    )

    with open(
        os.path.join(
            RESULT_DIR,
            "lstm_attention_test_report.txt"
        ),
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
        "LSTM + Attention Test Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        range(NUM_CLASSES),
        EMOTIONS,
        rotation=45
    )

    plt.yticks(
        range(NUM_CLASSES),
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
                str(matrix[row, col]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "lstm_attention_confusion_matrix.png"
        ),
        dpi=200
    )

    plt.close()

    return accuracy, macro_f1


# ============================================================
# MAIN
# ============================================================

def train():

    print("")
    print("=" * 70)
    print("TRAINING BIDIRECTIONAL LSTM + ATTENTION")
    print("=" * 70)

    if not check_dataset():
        return

    print("")
    print("[INFO] Loading train features...")
    x_train, y_train = load_split(
        TRAIN_DIR
    )

    print("[INFO] Loading validation features...")
    x_val, y_val = load_split(
        VAL_DIR
    )

    print("[INFO] Loading test features...")
    x_test, y_test = load_split(
        TEST_DIR
    )

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

    # ========================================================
    # NORMALIZE ORIGINAL CNN FEATURES
    # ========================================================

    x_train, x_val, x_test = normalize_features(
        x_train,
        x_val,
        x_test
    )

    # ========================================================
    # ADD TEMPORAL DELTA FEATURES
    # 256 CNN features + 256 delta features = 512
    # ========================================================

    x_train = add_temporal_deltas(
        x_train
    )

    x_val = add_temporal_deltas(
        x_val
    )

    x_test = add_temporal_deltas(
        x_test
    )

    print("")
    print(
        "V2 feature shapes:"
    )

    print(
        "Train:",
        x_train.shape
    )

    print(
        "Validation:",
        x_val.shape
    )

    print(
        "Test:",
        x_test.shape
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = get_class_weights(
        y_train
    )
    train_ds = make_tf_dataset(
        x_train,
        y_train,
        training=True
    )

    val_ds = make_tf_dataset(
        x_val,
        y_val,
        training=False
    )

    test_ds = make_tf_dataset(
        x_test,
        y_test,
        training=False
    )

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("")
    print(model.summary())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks()
    )

    save_history(
        history
    )

    model.save(
        FINAL_MODEL_PATH
    )

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

    evaluate_model(
        model,
        test_ds,
        y_test
    )

    print("")
    print("=" * 70)
    print("LSTM + ATTENTION TRAINING COMPLETED")
    print("=" * 70)

    print("")
    print("Best model:")
    print(BEST_MODEL_PATH)

    print("")
    print("Final model:")
    print(FINAL_MODEL_PATH)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    train()