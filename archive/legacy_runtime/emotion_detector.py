import os
import json
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

CNN_FEATURE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cnn_feature_extractor.keras"
)

TEMPORAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_best.keras"
)

MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_mean.npy"
)

STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_std.npy"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_classes.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 160

SEQUENCE_LENGTH = 20

FEATURE_DIM = 256

# تقریباً 10 نمونه در ثانیه
# پس 20 فریم ≈ حدود 2 ثانیه
SAMPLE_INTERVAL = 0.10

# بعد از کامل شدن Buffer،
# هر 2 نمونه یک Prediction جدید
PREDICT_EVERY_SAMPLES = 2

# برای جلوگیری از پرش درصدها
SMOOTHING_ALPHA = 0.35


DEFAULT_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


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
# EMOTION DETECTOR
# ============================================================

class EmotionDetector:

    def __init__(
        self,
        sample_interval=SAMPLE_INTERVAL,
        predict_every_samples=PREDICT_EVERY_SAMPLES,
        smoothing_alpha=SMOOTHING_ALPHA
    ):

        print("")
        print("=" * 70)
        print("LOADING EMOTION DETECTOR")
        print("=" * 70)

        self.sample_interval = float(
            sample_interval
        )

        self.predict_every_samples = int(
            predict_every_samples
        )

        self.smoothing_alpha = float(
            smoothing_alpha
        )

        self._check_files()

        # ====================================================
        # CLASS NAMES
        # ====================================================

        self.classes = self._load_classes()

        print(
            "[OK] Classes:",
            self.classes
        )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        self.feature_mean = np.load(
            MEAN_PATH
        ).astype(
            np.float32
        )

        self.feature_std = np.load(
            STD_PATH
        ).astype(
            np.float32
        )

        self.feature_std = np.where(
            self.feature_std < 1e-6,
            1e-6,
            self.feature_std
        )

        print(
            "[OK] Normalization loaded"
        )

        print(
            "Mean shape:",
            self.feature_mean.shape
        )

        print(
            "Std shape:",
            self.feature_std.shape
        )

        # ====================================================
        # CNN FEATURE EXTRACTOR
        # ====================================================

        print("")
        print(
            "[INFO] Loading CNN feature extractor..."
        )

        self.cnn_model = (
            tf.keras.models.load_model(
                CNN_FEATURE_MODEL_PATH,
                compile=False
            )
        )

        print(
            "[OK] CNN feature extractor loaded"
        )

        print(
            "CNN output shape:",
            self.cnn_model.output_shape
        )

        # ====================================================
        # LSTM + ATTENTION
        # ====================================================

        print("")
        print(
            "[INFO] Loading LSTM V1 Rebuild..."
        )

        custom_objects = {
            "TemporalAttention":
                TemporalAttention,

            "EmotionAI>TemporalAttention":
                TemporalAttention
        }

        self.temporal_model = (
            tf.keras.models.load_model(
                TEMPORAL_MODEL_PATH,
                custom_objects=custom_objects,
                compile=False
            )
        )

        print(
            "[OK] LSTM V1 loaded"
        )

        print(
            "LSTM input shape:",
            self.temporal_model.input_shape
        )

        print(
            "LSTM output shape:",
            self.temporal_model.output_shape
        )

        # ====================================================
        # ROLLING BUFFER
        # ====================================================

        self.feature_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.last_sample_time = 0.0

        self.samples_since_prediction = 0

        self.latest_probabilities = None

        self.smoothed_probabilities = None

        self.latest_label = None

        self.latest_confidence = 0.0

        print("")
        print("=" * 70)
        print("EMOTION DETECTOR READY")
        print("=" * 70)
        print("")


    # ========================================================
    # FILE CHECK
    # ========================================================

    def _check_files(
        self
    ):

        required_files = [

            CNN_FEATURE_MODEL_PATH,

            TEMPORAL_MODEL_PATH,

            MEAN_PATH,

            STD_PATH

        ]

        for path in required_files:

            if not os.path.isfile(
                path
            ):

                raise FileNotFoundError(
                    f"Required file not found:\n{path}"
                )


    # ========================================================
    # LOAD CLASS NAMES
    # ========================================================

    def _load_classes(
        self
    ):

        if not os.path.isfile(
            CLASS_PATH
        ):

            return DEFAULT_CLASSES.copy()


        try:

            with open(
                CLASS_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(
                    file
                )


            if isinstance(
                data,
                list
            ):

                return data


            if isinstance(
                data,
                dict
            ):

                if "classes" in data:

                    return data[
                        "classes"
                    ]


        except Exception as error:

            print(
                "[WARNING] Could not load class file:",
                error
            )


        return DEFAULT_CLASSES.copy()


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self
    ):

        self.feature_buffer.clear()

        self.last_sample_time = 0.0

        self.samples_since_prediction = 0

        self.latest_probabilities = None

        self.smoothed_probabilities = None

        self.latest_label = None

        self.latest_confidence = 0.0


    # ========================================================
    # PREPROCESS + CNN FEATURE
    # ========================================================

    def _extract_feature(
        self,
        face_bgr
    ):

        resized = cv2.resize(
            face_bgr,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        # ====================================================
        # مهم:
        #
        # در Featureهای LSTM V1، Sequenceهای OpenCV مستقیماً
        # وارد CNN Feature Extractor شده بودند.
        #
        # بنابراین اینجا عمداً BGR -> RGB انجام نمی‌دهیم
        # تا شرایط Real-Time با Training همان LSTM یکی باشد.
        # ====================================================

        image = resized.astype(
            np.float32
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        feature = self.cnn_model.predict(
            image,
            verbose=0
        )[0]

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).reshape(-1)

        if feature.shape[0] != FEATURE_DIM:

            raise ValueError(
                "CNN feature dimension mismatch. "
                f"Expected {FEATURE_DIM}, "
                f"found {feature.shape}"
            )

        return feature


    # ========================================================
    # TEMPORAL PREDICTION
    # ========================================================

    def _predict_temporal(
        self
    ):

        sequence = np.stack(
            list(
                self.feature_buffer
            ),
            axis=0
        ).astype(
            np.float32
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_DIM
        ):

            raise ValueError(
                "Temporal sequence shape error: "
                f"{sequence.shape}"
            )

        # ====================================================
        # همان Normalization مربوط به V1
        # ====================================================

        sequence = (
            sequence
            - self.feature_mean
        ) / self.feature_std

        sequence = np.expand_dims(
            sequence,
            axis=0
        )

        probabilities = (
            self.temporal_model.predict(
                sequence,
                verbose=0
            )[0]
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32
        )

        # ====================================================
        # Probability smoothing
        # ====================================================

        if (
            self.smoothed_probabilities
            is None
        ):

            self.smoothed_probabilities = (
                probabilities.copy()
            )

        else:

            alpha = (
                self.smoothing_alpha
            )

            self.smoothed_probabilities = (

                alpha
                * probabilities

                +

                (
                    1.0
                    - alpha
                )
                * self.smoothed_probabilities
            )


        total = np.sum(
            self.smoothed_probabilities
        )

        if total > 0:

            self.smoothed_probabilities = (
                self.smoothed_probabilities
                / total
            )


        self.latest_probabilities = (
            self.smoothed_probabilities.copy()
        )


        best_index = int(
            np.argmax(
                self.latest_probabilities
            )
        )


        self.latest_label = (
            self.classes[
                best_index
            ]
        )


        self.latest_confidence = float(
            self.latest_probabilities[
                best_index
            ]
        )


    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        face_bgr
    ):

        current_time = time.monotonic()

        # ====================================================
        # Temporal sampling
        # ====================================================

        if (
            current_time
            - self.last_sample_time
            <
            self.sample_interval
        ):

            return self.get_state()


        self.last_sample_time = (
            current_time
        )


        # ====================================================
        # CNN
        # ====================================================

        feature = self._extract_feature(
            face_bgr
        )


        # ====================================================
        # Rolling Buffer
        # ====================================================

        self.feature_buffer.append(
            feature
        )


        self.samples_since_prediction += 1


        # ====================================================
        # هنوز 20 فریم نداریم
        # ====================================================

        if len(
            self.feature_buffer
        ) < SEQUENCE_LENGTH:

            return self.get_state()


        # ====================================================
        # اولین Prediction یا Update دوره‌ای
        # ====================================================

        should_predict = (

            self.latest_probabilities
            is None

            or

            self.samples_since_prediction
            >=
            self.predict_every_samples

        )


        if should_predict:

            self._predict_temporal()

            self.samples_since_prediction = 0


        return self.get_state()


    # ========================================================
    # STATE
    # ========================================================

    def get_state(
        self
    ):

        collected = len(
            self.feature_buffer
        )

        ready = (

            collected
            >=
            SEQUENCE_LENGTH

            and

            self.latest_probabilities
            is not None
        )


        probability_dict = {}


        if (
            self.latest_probabilities
            is not None
        ):

            for index, emotion in enumerate(
                self.classes
            ):

                probability_dict[
                    emotion
                ] = float(

                    self.latest_probabilities[
                        index
                    ]

                )


        return {

            "ready":
                ready,

            "collected":
                collected,

            "required":
                SEQUENCE_LENGTH,

            "label":
                self.latest_label,

            "confidence":
                self.latest_confidence,

            "probabilities":
                probability_dict
        }


# ============================================================
# TEST MODEL LOADING
# ============================================================

if __name__ == "__main__":

    detector = EmotionDetector()

    print("")
    print(
        "All final emotion models loaded successfully."
    )

    print("")
    print(
        "Rolling Buffer:"
    )

    print(
        f"0 / {SEQUENCE_LENGTH}"
    )

    print("")
    print(
        "Ready for webcam integration."
    )