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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

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

V1_CLASS_PATH = os.path.join(
    MODEL_DIR,
    "lstm_attention_v1_classes.json"
)

ADAPTER_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sad_neutral_adapter.keras"
)

ADAPTER_CLASS_PATH = os.path.join(
    MODEL_DIR,
    "sad_neutral_adapter_classes.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 160
SEQUENCE_LENGTH = 20
FEATURE_DIM = 256

SAMPLE_INTERVAL = 0.12
PREDICT_EVERY_SAMPLES = 1

SMOOTHING_ALPHA = 0.70

# Adapter is only trusted when it is clearly confident.
ADAPTER_CONFIDENCE_THRESHOLD = 0.70

# Do not let the adapter act when Sad+Neutral have essentially
# no probability mass in the original V1 prediction.
ADAPTER_MIN_GROUP_MASS = 0.15

SWITCH_MIN_CONFIDENCE = 0.40
SWITCH_MIN_MARGIN = 0.08


DEFAULT_V1_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

FINAL_CLASSES = [
    "angry",
    "happy",
    "surprise",
    "sad",
    "neutral",
]

DEFAULT_ADAPTER_CLASSES = [
    "neutral",
    "sad",
]


# ============================================================
# CUSTOM TEMPORAL ATTENTION
# ============================================================

@tf.keras.utils.register_keras_serializable(package="EmotionAI")
class TemporalAttention(tf.keras.layers.Layer):

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)

        self.units = units

        self.score_dense = tf.keras.layers.Dense(
            units,
            activation="tanh"
        )

        self.weight_dense = tf.keras.layers.Dense(
            1
        )

    def call(self, inputs):
        scores = self.score_dense(inputs)
        scores = self.weight_dense(scores)

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
# 5-CLASS DETECTOR
# ============================================================

class EmotionDetector5Class:

    def __init__(
        self,
        sample_interval=SAMPLE_INTERVAL,
        predict_every_samples=PREDICT_EVERY_SAMPLES,
        smoothing_alpha=SMOOTHING_ALPHA,
        adapter_confidence_threshold=ADAPTER_CONFIDENCE_THRESHOLD,
    ):

        print("")
        print("=" * 72)
        print("LOADING FINAL 5-CLASS EMOTION DETECTOR")
        print("=" * 72)

        self.sample_interval = float(
            sample_interval
        )

        self.predict_every_samples = int(
            predict_every_samples
        )

        self.smoothing_alpha = float(
            smoothing_alpha
        )

        self.adapter_confidence_threshold = float(
            adapter_confidence_threshold
        )

        self._check_files()

        self.v1_classes = self._load_classes(
            V1_CLASS_PATH,
            DEFAULT_V1_CLASSES
        )

        self.adapter_classes = self._load_classes(
            ADAPTER_CLASS_PATH,
            DEFAULT_ADAPTER_CLASSES
        )

        if set(self.adapter_classes) != set(
            DEFAULT_ADAPTER_CLASSES
        ):
            raise ValueError(
                "Sad/Neutral adapter must contain only "
                "neutral and sad classes. "
                f"Found: {self.adapter_classes}"
            )

        self.v1_index = {
            name: index
            for index, name in enumerate(
                self.v1_classes
            )
        }

        required_v1 = set(
            DEFAULT_V1_CLASSES
        )

        if set(self.v1_classes) != required_v1:
            raise ValueError(
                "Unexpected V1 classes: "
                f"{self.v1_classes}"
            )

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
            "[INFO] Loading CNN feature extractor..."
        )

        self.cnn_model = (
            tf.keras.models.load_model(
                CNN_FEATURE_MODEL_PATH,
                compile=False
            )
        )

        print(
            "[OK] CNN output:",
            self.cnn_model.output_shape
        )

        print(
            "[INFO] Loading original LSTM V1..."
        )

        custom_objects = {
            "TemporalAttention":
                TemporalAttention,
            "EmotionAI>TemporalAttention":
                TemporalAttention,
        }

        self.temporal_model = (
            tf.keras.models.load_model(
                TEMPORAL_MODEL_PATH,
                custom_objects=custom_objects,
                compile=False
            )
        )

        print(
            "[OK] LSTM V1:",
            self.temporal_model.input_shape,
            "->",
            self.temporal_model.output_shape
        )

        print(
            "[INFO] Loading Sad/Neutral adapter..."
        )

        self.adapter_model = (
            tf.keras.models.load_model(
                ADAPTER_MODEL_PATH,
                compile=False
            )
        )

        print(
            "[OK] Sad/Neutral adapter:",
            self.adapter_model.input_shape,
            "->",
            self.adapter_model.output_shape
        )

        self.feature_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.last_sample_time = 0.0
        self.samples_since_prediction = 0

        self.latest_probabilities = None
        self.smoothed_probabilities = None

        self.latest_label = None
        self.latest_confidence = 0.0
        self.latest_margin = 0.0
        self.latest_reliability = "WARMUP"

        self.adapter_probabilities = None
        self.adapter_active = False
        self.adapter_confidence = 0.0

        self.base_7_probabilities = None
        self.merged_5_probabilities = None

        print("")
        print(
            "Final classes:",
            FINAL_CLASSES
        )

        print(
            "Merge rule: Disgust -> Sad"
        )

        print(
            "Merge rule: Fear -> Surprise"
        )

        print(
            "Adapter: only Sad <-> Neutral"
        )

        print(
            f"Adapter confidence threshold: "
            f"{self.adapter_confidence_threshold:.2f}"
        )

        print("")
        print("=" * 72)
        print("FINAL 5-CLASS DETECTOR READY")
        print("=" * 72)
        print("")

    # ========================================================
    # HELPERS
    # ========================================================

    def _check_files(self):

        required_files = [
            CNN_FEATURE_MODEL_PATH,
            TEMPORAL_MODEL_PATH,
            MEAN_PATH,
            STD_PATH,
            ADAPTER_MODEL_PATH,
        ]

        for path in required_files:

            if not os.path.isfile(path):

                raise FileNotFoundError(
                    f"Required file not found:\n{path}"
                )

    @staticmethod
    def _load_classes(
        path,
        default_classes
    ):

        if not os.path.isfile(path):
            return default_classes.copy()

        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            if isinstance(
                data,
                list
            ):
                return [
                    str(item).lower()
                    for item in data
                ]

            if (
                isinstance(
                    data,
                    dict
                )
                and
                "classes" in data
            ):
                return [
                    str(item).lower()
                    for item in data[
                        "classes"
                    ]
                ]

        except Exception as error:

            print(
                "[WARNING] Could not load class file:",
                error
            )

        return default_classes.copy()

    @staticmethod
    def _normalize_probs(
        probabilities
    ):

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32
        )

        probabilities = np.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        probabilities = np.maximum(
            probabilities,
            0.0
        )

        total = float(
            np.sum(
                probabilities
            )
        )

        if total <= 0:

            return np.full(
                len(
                    probabilities
                ),
                1.0
                / len(
                    probabilities
                ),
                dtype=np.float32
            )

        return (
            probabilities
            / total
        )

    @staticmethod
    def _top_stats(
        probabilities
    ):

        order = np.argsort(
            probabilities
        )[::-1]

        top1 = int(
            order[0]
        )

        top2 = int(
            order[1]
        )

        confidence = float(
            probabilities[
                top1
            ]
        )

        margin = float(
            probabilities[
                top1
            ]
            -
            probabilities[
                top2
            ]
        )

        return (
            top1,
            top2,
            confidence,
            margin
        )

    @staticmethod
    def _reliability(
        confidence,
        margin
    ):

        if (
            confidence >= 0.60
            and
            margin >= 0.15
        ):
            return "HIGH"

        if (
            confidence >= 0.45
            and
            margin >= 0.08
        ):
            return "MEDIUM"

        return "LOW"

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.feature_buffer.clear()

        self.last_sample_time = 0.0

        self.samples_since_prediction = 0

        self.latest_probabilities = None
        self.smoothed_probabilities = None

        self.latest_label = None
        self.latest_confidence = 0.0
        self.latest_margin = 0.0
        self.latest_reliability = "WARMUP"

        self.adapter_probabilities = None
        self.adapter_active = False
        self.adapter_confidence = 0.0

        self.base_7_probabilities = None
        self.merged_5_probabilities = None

    # ========================================================
    # CNN FEATURE EXTRACTION
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

        # Intentionally keep BGR for V1 compatibility.
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
    # MERGE 7 CLASSES -> 5 CLASSES
    # ========================================================

    def _merge_v1_to_5(
        self,
        base_7
    ):

        angry = base_7[
            self.v1_index[
                "angry"
            ]
        ]

        happy = base_7[
            self.v1_index[
                "happy"
            ]
        ]

        surprise = (
            base_7[
                self.v1_index[
                    "surprise"
                ]
            ]
            +
            base_7[
                self.v1_index[
                    "fear"
                ]
            ]
        )

        sad = (
            base_7[
                self.v1_index[
                    "sad"
                ]
            ]
            +
            base_7[
                self.v1_index[
                    "disgust"
                ]
            ]
        )

        neutral = base_7[
            self.v1_index[
                "neutral"
            ]
        ]

        merged = np.asarray(
            [
                angry,
                happy,
                surprise,
                sad,
                neutral,
            ],
            dtype=np.float32
        )

        return self._normalize_probs(
            merged
        )

    # ========================================================
    # SAFE SAD/NEUTRAL ADAPTER
    # ========================================================

    def _apply_adapter_if_confident(
        self,
        merged_5,
        adapter_probs
    ):

        final_probs = (
            merged_5.copy()
        )

        sad_index = FINAL_CLASSES.index(
            "sad"
        )

        neutral_index = FINAL_CLASSES.index(
            "neutral"
        )

        group_mass = float(
            final_probs[
                sad_index
            ]
            +
            final_probs[
                neutral_index
            ]
        )

        adapter_top = float(
            np.max(
                adapter_probs
            )
        )

        use_adapter = (
            adapter_top
            >=
            self.adapter_confidence_threshold
            and
            group_mass
            >=
            ADAPTER_MIN_GROUP_MASS
        )

        if not use_adapter:

            return (
                final_probs,
                False,
                group_mass,
                adapter_top
            )

        adapter_map = {
            emotion:
                float(
                    adapter_probs[
                        index
                    ]
                )
            for index, emotion in enumerate(
                self.adapter_classes
            )
        }

        final_probs[
            neutral_index
        ] = (
            group_mass
            *
            adapter_map[
                "neutral"
            ]
        )

        final_probs[
            sad_index
        ] = (
            group_mass
            *
            adapter_map[
                "sad"
            ]
        )

        # Angry / Happy / Surprise are untouched.
        return (
            self._normalize_probs(
                final_probs
            ),
            True,
            group_mass,
            adapter_top
        )

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

        normalized = (
            sequence
            - self.feature_mean
        ) / self.feature_std

        normalized = np.expand_dims(
            normalized,
            axis=0
        )

        # ----------------------------------------------------
        # Original V1 output
        # ----------------------------------------------------

        base_7 = (
            self.temporal_model.predict(
                normalized,
                verbose=0
            )[0]
        )

        base_7 = self._normalize_probs(
            base_7
        )

        # ----------------------------------------------------
        # Merge to 5 classes
        # ----------------------------------------------------

        merged_5 = self._merge_v1_to_5(
            base_7
        )

        # ----------------------------------------------------
        # Sad/Neutral adapter
        # ----------------------------------------------------

        adapter_probs = (
            self.adapter_model.predict(
                normalized,
                verbose=0
            )[0]
        )

        adapter_probs = self._normalize_probs(
            adapter_probs
        )

        (
            final_raw,
            adapter_active,
            _group_mass,
            adapter_confidence
        ) = self._apply_adapter_if_confident(
            merged_5,
            adapter_probs
        )

        self.base_7_probabilities = (
            base_7.copy()
        )

        self.merged_5_probabilities = (
            merged_5.copy()
        )

        self.adapter_probabilities = (
            adapter_probs.copy()
        )

        self.adapter_active = (
            adapter_active
        )

        self.adapter_confidence = (
            adapter_confidence
        )

        # ----------------------------------------------------
        # Adaptive smoothing
        # ----------------------------------------------------

        (
            raw_top1,
            _,
            raw_confidence,
            raw_margin
        ) = self._top_stats(
            final_raw
        )

        raw_label = FINAL_CLASSES[
            raw_top1
        ]

        if (
            self.smoothed_probabilities
            is None
        ):

            smoothed = (
                final_raw.copy()
            )

        else:

            strong_switch = (
                self.latest_label
                is not None
                and
                raw_label
                !=
                self.latest_label
                and
                raw_confidence
                >=
                SWITCH_MIN_CONFIDENCE
                and
                raw_margin
                >=
                SWITCH_MIN_MARGIN
            )

            if strong_switch:

                smoothed = (
                    final_raw.copy()
                )

            else:

                alpha = (
                    self.smoothing_alpha
                )

                smoothed = (
                    alpha
                    * final_raw
                    +
                    (
                        1.0
                        - alpha
                    )
                    * self.smoothed_probabilities
                )

        smoothed = self._normalize_probs(
            smoothed
        )

        self.smoothed_probabilities = (
            smoothed.copy()
        )

        self.latest_probabilities = (
            smoothed.copy()
        )

        (
            best_index,
            _,
            confidence,
            margin
        ) = self._top_stats(
            self.latest_probabilities
        )

        self.latest_label = (
            FINAL_CLASSES[
                best_index
            ]
        )

        self.latest_confidence = (
            confidence
        )

        self.latest_margin = (
            margin
        )

        self.latest_reliability = (
            self._reliability(
                confidence,
                margin
            )
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        face_bgr
    ):

        current_time = (
            time.monotonic()
        )

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

        feature = self._extract_feature(
            face_bgr
        )

        self.feature_buffer.append(
            feature
        )

        self.samples_since_prediction += 1

        if len(
            self.feature_buffer
        ) < SEQUENCE_LENGTH:

            return self.get_state()

        if (
            self.latest_probabilities
            is None
            or
            self.samples_since_prediction
            >=
            self.predict_every_samples
        ):

            self._predict_temporal()

            self.samples_since_prediction = 0

        return self.get_state()

    # ========================================================
    # PUBLIC STATE
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

        final_dict = {}

        if (
            self.latest_probabilities
            is not None
        ):

            final_dict = {
                emotion:
                    float(
                        self.latest_probabilities[
                            index
                        ]
                    )
                for index, emotion in enumerate(
                    FINAL_CLASSES
                )
            }

        adapter_dict = {}

        if (
            self.adapter_probabilities
            is not None
        ):

            adapter_dict = {
                emotion:
                    float(
                        self.adapter_probabilities[
                            index
                        ]
                    )
                for index, emotion in enumerate(
                    self.adapter_classes
                )
            }

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

            "margin":
                self.latest_margin,

            "reliability":
                self.latest_reliability,

            "probabilities":
                final_dict,

            "adapter_probabilities":
                adapter_dict,

            "adapter_active":
                self.adapter_active,

            "adapter_confidence":
                self.adapter_confidence,

            "classes":
                FINAL_CLASSES.copy()
        }


# ============================================================
# LOAD TEST
# ============================================================

if __name__ == "__main__":

    detector = EmotionDetector5Class()

    print(
        "All final 5-class models loaded successfully."
    )

    print(
        f"Rolling Buffer: 0 / {SEQUENCE_LENGTH}"
    )

    print(
        "Ready for webcam integration."
    )
