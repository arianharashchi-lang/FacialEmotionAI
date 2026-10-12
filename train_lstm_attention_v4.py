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
# EMOTIONS
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
# WEAK CLASSES
# ============================================================

FEAR_INDEX = EMOTIONS.index(
    "fear"
)

SAD_INDEX = EMOTIONS.index(
    "sad"
)

SURPRISE_INDEX = EMOTIONS.index(
    "surprise"
)

HARD_CLASSES = [
    FEAR_INDEX,
    SAD_INDEX,
    SURPRISE_INDEX
]


# ============================================================
# DATA SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20

FEATURE_DIM = 256


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 3e-4

FOCAL_GAMMA = 1.5

# فقط 50 درصد نمونه اضافه برای سه کلاس سخت
HARD_AUGMENT_RATIO = 0.50

FEATURE_NOISE_STD = 0.025


# ============================================================
# OUTPUT PATHS
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v4_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v4_final.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v4_classes.json"
)

MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v4_feature_mean.npy"
)

STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v4_feature_std.npy"
)

ALPHA_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v4_focal_alpha.json"
)

REPORT_PATH = os.path.join(
    RESULT_DIR,
    "lstm_attention_v4_test_report.txt"
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "lstm_attention_v4_metrics.json"
)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print("")
    print("=" * 70)
    print("CHECKING V4 DATASET")
    print("=" * 70)

    ok = True

    for split_name, split_dir in [

        ("TRAIN", TRAIN_DIR),
        ("VALIDATION", VAL_DIR),
        ("TEST", TEST_DIR)

    ]:

        print("")
        print(split_name)
        print("-" * 40)

        total = 0

        if not os.path.isdir(
            split_dir
        ):

            print(
                "[ERROR] Missing:"
            )

            print(
                split_dir
            )

            ok = False
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
                    "[ERROR] Missing emotion:",
                    emotion
                )

                ok = False
                continue


            count = len([
                f
                for f in os.listdir(
                    emotion_dir
                )
                if f.lower().endswith(
                    ".npy"
                )
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

                    f"Invalid feature shape\n"
                    f"File: {path}\n"
                    f"Found: {features.shape}\n"
                    f"Expected: "
                    f"({SEQUENCE_LENGTH}, "
                    f"{FEATURE_DIM})"

                )


            x_data.append(
                features.astype(
                    np.float32
                )
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


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_train_validation(
    x_train,
    x_val
):

    print("")
    print("=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)


    # فقط TRAIN
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
        x_train
        - feature_mean
    ) / feature_std


    x_val = (
        x_val
        - feature_mean
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
        "[SUCCESS] Mean/Std saved."
    )


    return (
        x_train.astype(
            np.float32
        ),

        x_val.astype(
            np.float32
        ),

        feature_mean.astype(
            np.float32
        ),

        feature_std.astype(
            np.float32
        )
    )


# ============================================================
# HARD CLASS FEATURE AUGMENTATION
# ============================================================

def augment_hard_classes(
    x_train,
    y_train
):

    print("")
    print("=" * 70)
    print("HARD CLASS TEMPORAL AUGMENTATION")
    print("=" * 70)


    rng = np.random.default_rng(
        SEED
    )


    augmented_x = [
        x_train
    ]

    augmented_y = [
        y_train
    ]


    for class_index in HARD_CLASSES:

        indices = np.where(
            y_train == class_index
        )[0]


        original_count = len(
            indices
        )


        augment_count = int(
            original_count
            * HARD_AUGMENT_RATIO
        )


        selected_indices = rng.choice(
            indices,
            size=augment_count,
            replace=False
        )


        new_samples = []


        for index in selected_indices:

            sample = x_train[
                index
            ].copy()


            # =================================================
            # Feature noise
            # =================================================

            noise = rng.normal(

                loc=0.0,

                scale=FEATURE_NOISE_STD,

                size=sample.shape

            ).astype(
                np.float32
            )


            sample = (
                sample
                + noise
            )


            # =================================================
            # Mild temporal smoothing
            #
            # حرکت زمانی حفظ می‌شود ولی مدل کمتر
            # به یک Frame خاص وابسته می‌شود.
            # =================================================

            if rng.random() < 0.5:

                smoothed = sample.copy()

                smoothed[1:] = (
                    0.85
                    * sample[1:]
                    +
                    0.15
                    * sample[:-1]
                )

                sample = smoothed


            # =================================================
            # Random single-frame masking
            # =================================================

            if rng.random() < 0.35:

                frame_index = int(
                    rng.integers(
                        1,
                        SEQUENCE_LENGTH - 1
                    )
                )


                sample[
                    frame_index
                ] = (

                    sample[
                        frame_index - 1
                    ]

                    +

                    sample[
                        frame_index + 1
                    ]

                ) / 2.0


            new_samples.append(
                sample.astype(
                    np.float32
                )
            )


        if new_samples:

            new_samples = np.asarray(
                new_samples,
                dtype=np.float32
            )


            new_labels = np.full(

                len(
                    new_samples
                ),

                class_index,

                dtype=np.int32

            )


            augmented_x.append(
                new_samples
            )

            augmented_y.append(
                new_labels
            )


        print(

            f"{EMOTIONS[class_index]:10s}: "
            f"{original_count} + "
            f"{augment_count}"

        )


    x_augmented = np.concatenate(
        augmented_x,
        axis=0
    )


    y_augmented = np.concatenate(
        augmented_y,
        axis=0
    )


    # ========================================================
    # Shuffle
    # ========================================================

    permutation = rng.permutation(
        len(
            x_augmented
        )
    )


    x_augmented = x_augmented[
        permutation
    ]


    y_augmented = y_augmented[
        permutation
    ]


    print("")
    print(
        "Original training samples:",
        len(
            x_train
        )
    )

    print(
        "Augmented training samples:",
        len(
            x_augmented
        )
    )


    return (
        x_augmented,
        y_augmented
    )


# ============================================================
# FOCAL ALPHA
# ============================================================

def calculate_focal_alpha(
    y_train
):

    # ========================================================
    # پایه بر اساس class balance
    # ========================================================

    weights = compute_class_weight(

        class_weight="balanced",

        classes=np.arange(
            NUM_CLASSES
        ),

        y=y_train

    )


    # نرم‌تر کردن وزن‌های پایه
    weights = np.sqrt(
        weights
    )


    # ========================================================
    # Boost ملایم سه کلاس ضعیف
    # ========================================================

    boosters = np.ones(
        NUM_CLASSES,
        dtype=np.float32
    )


    boosters[
        FEAR_INDEX
    ] = 1.15


    boosters[
        SAD_INDEX
    ] = 1.20


    boosters[
        SURPRISE_INDEX
    ] = 1.30


    alpha = (
        weights
        * boosters
    )


    # میانگین وزن = 1
    alpha = (
        alpha
        / np.mean(alpha)
    )


    print("")
    print("=" * 70)
    print("V4 FOCAL ALPHA")
    print("=" * 70)


    alpha_dict = {}


    for index, value in enumerate(
        alpha
    ):

        alpha_dict[
            EMOTIONS[index]
        ] = float(
            value
        )


        print(

            f"{EMOTIONS[index]:10s}: "
            f"{value:.4f}"

        )


    with open(

        ALPHA_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(
            alpha_dict,
            file,
            indent=4
        )


    return alpha.astype(
        np.float32
    )


# ============================================================
# SPARSE FOCAL LOSS
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="EmotionAI"
)
class SparseFocalLoss(
    tf.keras.losses.Loss
):

    def __init__(
        self,
        alpha,
        gamma=1.5,
        name="sparse_focal_loss",
        **kwargs
    ):

        super().__init__(
            name=name,
            **kwargs
        )

        self.alpha = [
            float(x)
            for x in alpha
        ]

        self.gamma = float(
            gamma
        )


    def call(
        self,
        y_true,
        y_pred
    ):

        y_true = tf.cast(
            tf.reshape(
                y_true,
                [-1]
            ),
            tf.int32
        )


        y_pred = tf.clip_by_value(

            y_pred,

            tf.keras.backend.epsilon(),

            1.0
            - tf.keras.backend.epsilon()

        )


        indices = tf.stack(

            [

                tf.range(
                    tf.shape(
                        y_true
                    )[0]
                ),

                y_true

            ],

            axis=1

        )


        p_t = tf.gather_nd(
            y_pred,
            indices
        )


        alpha_tensor = tf.constant(
            self.alpha,
            dtype=tf.float32
        )


        alpha_t = tf.gather(
            alpha_tensor,
            y_true
        )


        focal_factor = tf.pow(

            1.0 - p_t,

            self.gamma

        )


        loss = (

            -alpha_t
            * focal_factor
            * tf.math.log(
                p_t
            )

        )


        return tf.reduce_mean(
            loss
        )


    def get_config(
        self
    ):

        config = super().get_config()


        config.update({

            "alpha":
                self.alpha,

            "gamma":
                self.gamma

        })


        return config


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


        self.hidden_dense = (
            tf.keras.layers.Dense(
                units,
                activation="tanh"
            )
        )


        self.score_dense = (
            tf.keras.layers.Dense(
                1
            )
        )


    def call(
        self,
        inputs
    ):

        hidden = self.hidden_dense(
            inputs
        )


        scores = self.score_dense(
            hidden
        )


        weights = tf.nn.softmax(
            scores,
            axis=1
        )


        context = tf.reduce_sum(

            weights
            * inputs,

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
# BUILD V4 MODEL
# ============================================================

def build_model():

    print("")
    print("=" * 70)
    print("BUILDING LSTM ATTENTION V4")
    print("=" * 70)


    inputs = tf.keras.Input(

        shape=(
            SEQUENCE_LENGTH,
            FEATURE_DIM
        ),

        name="sequence_features"

    )


    # ========================================================
    # Mild regularization
    # ========================================================

    x = tf.keras.layers.GaussianNoise(

        0.015,

        name="feature_noise"

    )(inputs)


    # ========================================================
    # BiLSTM 1
    # ========================================================

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            128,

            return_sequences=True,

            dropout=0.25,

            recurrent_dropout=0.10

        ),

        name="bilstm_1"

    )(x)


    # ========================================================
    # BiLSTM 2
    # ========================================================

    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(

            64,

            return_sequences=True,

            dropout=0.25,

            recurrent_dropout=0.10

        ),

        name="bilstm_2"

    )(x)


    x = tf.keras.layers.LayerNormalization(

        name="temporal_normalization"

    )(x)


    # ========================================================
    # ATTENTION CONTEXT
    # ========================================================

    attention_context = TemporalAttention(

        units=64,

        name="temporal_attention"

    )(x)


    # ========================================================
    # PEAK EXPRESSION FEATURE
    #
    # برای Expressionهای کوتاه و شدید مثل Surprise
    # ========================================================

    max_context = (
        tf.keras.layers.GlobalMaxPooling1D(

            name="temporal_peak_pool"

        )(x)
    )


    # ========================================================
    # COMBINE
    # ========================================================

    context = tf.keras.layers.Concatenate(

        name="attention_peak_fusion"

    )([

        attention_context,

        max_context

    ])


    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = tf.keras.layers.Dense(

        128,

        activation="relu",

        kernel_regularizer=(
            tf.keras.regularizers.l2(
                2e-4
            )
        ),

        name="dense_128"

    )(context)


    x = tf.keras.layers.LayerNormalization(

        name="classifier_normalization"

    )(x)


    x = tf.keras.layers.Dropout(

        0.40,

        name="classifier_dropout"

    )(x)


    outputs = tf.keras.layers.Dense(

        NUM_CLASSES,

        activation="softmax",

        name="emotion_output"

    )(x)


    return tf.keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="Emotion_LSTM_Attention_V4"

    )


# ============================================================
# VALIDATION METRICS CALLBACK
# ============================================================

class ValidationEmotionMetrics(
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


        class_f1 = f1_score(

            self.y_val,

            predictions,

            average=None,

            labels=np.arange(
                NUM_CLASSES
            ),

            zero_division=0

        )


        fear_f1 = class_f1[
            FEAR_INDEX
        ]


        sad_f1 = class_f1[
            SAD_INDEX
        ]


        surprise_f1 = class_f1[
            SURPRISE_INDEX
        ]


        weak_f1 = np.mean([

            fear_f1,

            sad_f1,

            surprise_f1

        ])


        # ====================================================
        # انتخاب مدل:
        # 75% کیفیت کلی
        # 25% سه کلاس ضعیف
        # ====================================================

        v4_score = (

            0.75
            * macro_f1

            +

            0.25
            * weak_f1

        )


        logs[
            "val_macro_f1"
        ] = float(
            macro_f1
        )


        logs[
            "val_weak_f1"
        ] = float(
            weak_f1
        )


        logs[
            "val_v4_score"
        ] = float(
            v4_score
        )


        print("")

        print(
            "[VAL V4]"
        )


        print(

            f"Macro F1     : "
            f"{macro_f1:.4f}"

        )


        print(

            f"Fear F1      : "
            f"{fear_f1:.4f}"

        )


        print(

            f"Sad F1       : "
            f"{sad_f1:.4f}"

        )


        print(

            f"Surprise F1  : "
            f"{surprise_f1:.4f}"

        )


        print(

            f"Weak Avg F1  : "
            f"{weak_f1:.4f}"

        )


        print(

            f"V4 Score     : "
            f"{v4_score:.4f}"

        )


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(
    x_val,
    y_val
):

    validation_metrics = (
        ValidationEmotionMetrics(

            x_val,

            y_val

        )
    )


    checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(

            BEST_MODEL_PATH,

            monitor="val_v4_score",

            mode="max",

            save_best_only=True,

            verbose=1

        )
    )


    early_stopping = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_v4_score",

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


    return [

        # باید اول باشد
        validation_metrics,

        checkpoint,

        early_stopping,

        reduce_lr

    ]


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(
    history
):

    # Accuracy

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
        "LSTM Attention V4 Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "lstm_attention_v4_accuracy.png"

        ),

        dpi=200

    )


    plt.close()


    # Loss

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
        "LSTM Attention V4 Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_DIR,

            "lstm_attention_v4_loss.png"

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
        "FINAL TEST - V4 - UNSEEN ACTORS"
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


    class_f1 = f1_score(

        y_test,

        predictions,

        average=None,

        labels=np.arange(
            NUM_CLASSES
        ),

        zero_division=0

    )


    print("")

    print(
        f"Test Accuracy : "
        f"{accuracy:.4f}"
    )


    print(
        f"Macro F1      : "
        f"{macro_f1:.4f}"
    )


    print(
        f"Weighted F1   : "
        f"{weighted_f1:.4f}"
    )


    print("")
    print(
        "TARGET CLASS F1"
    )

    print(
        "-" * 40
    )


    print(

        f"Fear      : "
        f"{class_f1[FEAR_INDEX]:.4f}"

    )


    print(

        f"Sad       : "
        f"{class_f1[SAD_INDEX]:.4f}"

    )


    print(

        f"Surprise  : "
        f"{class_f1[SURPRISE_INDEX]:.4f}"

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

            f"Macro F1: "
            f"{macro_f1:.6f}\n"

        )

        file.write(

            f"Weighted F1: "
            f"{weighted_f1:.6f}\n"

        )

        file.write(

            f"Fear F1: "
            f"{class_f1[FEAR_INDEX]:.6f}\n"

        )

        file.write(

            f"Sad F1: "
            f"{class_f1[SAD_INDEX]:.6f}\n"

        )

        file.write(

            f"Surprise F1: "
            f"{class_f1[SURPRISE_INDEX]:.6f}\n\n"

        )

        file.write(
            report
        )


    metrics = {

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "fear_f1":
            float(
                class_f1[
                    FEAR_INDEX
                ]
            ),

        "sad_f1":
            float(
                class_f1[
                    SAD_INDEX
                ]
            ),

        "surprise_f1":
            float(
                class_f1[
                    SURPRISE_INDEX
                ]
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


    # Confusion Matrix

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
        "LSTM Attention V4 Confusion Matrix"
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

            "lstm_attention_v4_confusion_matrix.png"

        ),

        dpi=200

    )


    plt.close()


    return (
        accuracy,
        macro_f1,
        class_f1
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    print("")
    print("=" * 70)

    print(
        "LSTM + ATTENTION V4"
    )

    print(
        "TARGET: FEAR / SAD / SURPRISE"
    )

    print("=" * 70)


    if not check_dataset():

        return


    # ========================================================
    # LOAD TRAIN + VALIDATION ONLY
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


    print("")
    print(
        "Train:",
        x_train.shape
    )

    print(
        "Validation:",
        x_val.shape
    )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    (
        x_train,
        x_val,
        feature_mean,
        feature_std

    ) = normalize_train_validation(

        x_train,
        x_val

    )


    # ========================================================
    # ALPHA BEFORE AUGMENTATION
    #
    # بر اساس توزیع واقعی Dataset
    # ========================================================

    focal_alpha = calculate_focal_alpha(
        y_train
    )


    # ========================================================
    # AUGMENT TRAIN ONLY
    # ========================================================

    (
        x_train,
        y_train

    ) = augment_hard_classes(

        x_train,
        y_train

    )


    print("")
    print(
        "Final V4 Train shape:"
    )

    print(
        x_train.shape
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = build_model()


    focal_loss = SparseFocalLoss(

        alpha=focal_alpha,

        gamma=FOCAL_GAMMA

    )


    model.compile(

        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            )
        ),

        loss=focal_loss,

        metrics=[
            "accuracy"
        ]

    )


    model.summary()


    # ========================================================
    # TRAIN
    # ========================================================

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

        callbacks=get_callbacks(
            x_val,
            y_val
        ),

        verbose=1

    )


    # Restore best weights already handled
    # by EarlyStopping.

    model.save(
        FINAL_MODEL_PATH
    )


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


    save_history(
        history
    )


    # ========================================================
    # TEST IS LOADED ONLY NOW
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "LOADING TEST ONLY AFTER TRAINING"
    )

    print("=" * 70)


    x_test, y_test = load_split(
        TEST_DIR
    )


    x_test = (

        x_test
        - feature_mean

    ) / feature_std


    x_test = x_test.astype(
        np.float32
    )


    print(
        "Test:",
        x_test.shape
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    (
        accuracy,
        macro_f1,
        class_f1

    ) = evaluate_test(

        model,

        x_test,

        y_test

    )


    print("")
    print("=" * 70)

    print(
        "V4 TRAINING COMPLETED"
    )

    print("=" * 70)


    print("")
    print(
        "Best V4:"
    )

    print(
        BEST_MODEL_PATH
    )


    print("")
    print(
        "Final V4:"
    )

    print(
        FINAL_MODEL_PATH
    )


    print("")
    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )


    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )


    print("")
    print(
        "Weak emotion F1:"
    )


    print(
        "Fear:",
        f"{class_f1[FEAR_INDEX]:.4f}"
    )


    print(
        "Sad:",
        f"{class_f1[SAD_INDEX]:.4f}"
    )


    print(
        "Surprise:",
        f"{class_f1[SURPRISE_INDEX]:.4f}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train()