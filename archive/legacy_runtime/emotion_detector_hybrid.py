import os
import json
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CNN_FEATURE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_feature_extractor.keras")
TEMPORAL_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_attention_v1_best.keras")
MEAN_PATH = os.path.join(MODEL_DIR, "lstm_v1_feature_mean.npy")
STD_PATH = os.path.join(MODEL_DIR, "lstm_v1_feature_std.npy")
CLASS_PATH = os.path.join(MODEL_DIR, "lstm_attention_v1_classes.json")

WEAK_ADAPTER_PATH = os.path.join(MODEL_DIR, "weak_emotion_adapter.keras")
WEAK_CLASS_PATH = os.path.join(MODEL_DIR, "weak_emotion_adapter_classes.json")

IMAGE_SIZE = 160
SEQUENCE_LENGTH = 20
FEATURE_DIM = 256

SAMPLE_INTERVAL = 0.12
PREDICT_EVERY_SAMPLES = 1
SMOOTHING_ALPHA = 0.70

SWITCH_MIN_CONFIDENCE = 0.40
SWITCH_MIN_MARGIN = 0.08

DEFAULT_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

DEFAULT_WEAK_CLASSES = [
    "neutral",
    "sad",
    "fear",
]


@tf.keras.utils.register_keras_serializable(package="EmotionAI")
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(units, activation="tanh")
        self.weight_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        scores = self.score_dense(inputs)
        scores = self.weight_dense(scores)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * inputs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class EmotionDetector:
    """
    Hybrid detector:

    1) Original V1 still computes all 7 emotion probabilities.
    2) Angry / Disgust / Happy / Surprise are NEVER changed by the specialist.
    3) Only the probability mass already assigned by V1 to
       Neutral + Sad + Fear is redistributed by the weak-emotion adapter.

    Therefore the four strong-class raw probabilities remain exactly V1's.
    """

    def __init__(
        self,
        sample_interval=SAMPLE_INTERVAL,
        predict_every_samples=PREDICT_EVERY_SAMPLES,
        smoothing_alpha=SMOOTHING_ALPHA,
    ):
        print("\n" + "=" * 72)
        print("LOADING HYBRID EMOTION DETECTOR")
        print("V1 + NEUTRAL/SAD/FEAR SPECIALIST")
        print("=" * 72)

        self.sample_interval = float(sample_interval)
        self.predict_every_samples = int(predict_every_samples)
        self.smoothing_alpha = float(smoothing_alpha)

        self._check_files()

        self.classes = self._load_json_classes(
            CLASS_PATH,
            DEFAULT_CLASSES,
        )

        self.weak_classes = self._load_json_classes(
            WEAK_CLASS_PATH,
            DEFAULT_WEAK_CLASSES,
        )

        expected_weak = set(DEFAULT_WEAK_CLASSES)
        if set(self.weak_classes) != expected_weak:
            raise ValueError(
                "Weak adapter classes must be neutral/sad/fear. "
                f"Found: {self.weak_classes}"
            )

        self.feature_mean = np.load(MEAN_PATH).astype(np.float32)
        self.feature_std = np.load(STD_PATH).astype(np.float32)
        self.feature_std = np.where(
            self.feature_std < 1e-6,
            1e-6,
            self.feature_std
        )

        print("[INFO] Loading CNN feature extractor...")
        self.cnn_model = tf.keras.models.load_model(
            CNN_FEATURE_MODEL_PATH,
            compile=False,
        )
        print("[OK] CNN:", self.cnn_model.output_shape)

        print("[INFO] Loading original LSTM V1...")
        custom_objects = {
            "TemporalAttention": TemporalAttention,
            "EmotionAI>TemporalAttention": TemporalAttention,
        }
        self.temporal_model = tf.keras.models.load_model(
            TEMPORAL_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
        )
        print(
            "[OK] LSTM V1:",
            self.temporal_model.input_shape,
            "->",
            self.temporal_model.output_shape,
        )

        print("[INFO] Loading weak-emotion specialist...")
        self.weak_adapter = tf.keras.models.load_model(
            WEAK_ADAPTER_PATH,
            compile=False,
        )
        print(
            "[OK] Weak adapter:",
            self.weak_adapter.input_shape,
            "->",
            self.weak_adapter.output_shape,
        )

        if int(self.temporal_model.output_shape[-1]) != len(self.classes):
            raise ValueError(
                "V1 class count mismatch: "
                f"model={self.temporal_model.output_shape[-1]}, "
                f"classes={len(self.classes)}"
            )

        if int(self.weak_adapter.output_shape[-1]) != len(self.weak_classes):
            raise ValueError(
                "Weak adapter class count mismatch: "
                f"model={self.weak_adapter.output_shape[-1]}, "
                f"classes={len(self.weak_classes)}"
            )

        self.class_to_index = {
            name: index
            for index, name in enumerate(self.classes)
        }

        self.weak_indices = [
            self.class_to_index[name]
            for name in self.weak_classes
        ]

        self.strong_classes = [
            name
            for name in self.classes
            if name not in self.weak_classes
        ]

        print("")
        print("[LOCKED STRONG CLASSES]")
        print(self.strong_classes)
        print("")
        print("[SPECIALIST CLASSES]")
        print(self.weak_classes)

        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.last_sample_time = 0.0
        self.samples_since_prediction = 0

        self.latest_probabilities = None
        self.smoothed_probabilities = None

        self.base_probabilities = None
        self.hybrid_raw_probabilities = None
        self.adapter_probabilities = None
        self.weak_mass = 0.0

        self.latest_label = None
        self.latest_confidence = 0.0
        self.latest_margin = 0.0
        self.latest_reliability = "WARMUP"

        print("")
        print("=" * 72)
        print("HYBRID EMOTION DETECTOR READY")
        print("=" * 72)
        print("")

    def _check_files(self):
        required = [
            CNN_FEATURE_MODEL_PATH,
            TEMPORAL_MODEL_PATH,
            MEAN_PATH,
            STD_PATH,
            WEAK_ADAPTER_PATH,
        ]

        for path in required:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Required file not found:\n{path}"
                )

    @staticmethod
    def _load_json_classes(path, default_classes):
        if not os.path.isfile(path):
            return default_classes.copy()

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                return [str(x).lower() for x in data]

            if isinstance(data, dict) and "classes" in data:
                return [
                    str(x).lower()
                    for x in data["classes"]
                ]

        except Exception as exc:
            print(
                "[WARNING] Could not load classes from",
                path,
                ":",
                exc,
            )

        return default_classes.copy()

    def reset(self):
        self.feature_buffer.clear()
        self.last_sample_time = 0.0
        self.samples_since_prediction = 0

        self.latest_probabilities = None
        self.smoothed_probabilities = None

        self.base_probabilities = None
        self.hybrid_raw_probabilities = None
        self.adapter_probabilities = None
        self.weak_mass = 0.0

        self.latest_label = None
        self.latest_confidence = 0.0
        self.latest_margin = 0.0
        self.latest_reliability = "WARMUP"

    def _extract_feature(self, face_bgr):
        resized = cv2.resize(
            face_bgr,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # Keep BGR intentionally to match the V1 temporal feature pipeline.
        image = np.clip(
            resized,
            0,
            255
        ).astype(np.float32)[None, ...]

        feature = self.cnn_model.predict(
            image,
            verbose=0,
        )[0]

        feature = np.asarray(
            feature,
            dtype=np.float32,
        ).reshape(-1)

        if feature.shape[0] != FEATURE_DIM:
            raise ValueError(
                "CNN feature dimension mismatch: "
                f"expected {FEATURE_DIM}, got {feature.shape}"
            )

        return feature

    @staticmethod
    def _normalize_probability_vector(probabilities):
        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )
        probabilities = np.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        probabilities = np.maximum(
            probabilities,
            0.0,
        )

        total = float(
            probabilities.sum()
        )

        if total <= 0:
            return np.full(
                len(probabilities),
                1.0 / len(probabilities),
                dtype=np.float32,
            )

        return probabilities / total

    @staticmethod
    def _top_stats(probabilities):
        order = np.argsort(
            probabilities
        )[::-1]

        top1 = int(order[0])
        top2 = int(order[1])

        confidence = float(
            probabilities[top1]
        )

        margin = float(
            probabilities[top1]
            - probabilities[top2]
        )

        return (
            top1,
            top2,
            confidence,
            margin,
        )

    @staticmethod
    def _reliability(confidence, margin):
        if confidence >= 0.60 and margin >= 0.15:
            return "HIGH"

        if confidence >= 0.45 and margin >= 0.08:
            return "MEDIUM"

        return "LOW"

    def _make_hybrid_probabilities(
        self,
        base_probs,
        adapter_probs,
    ):
        """
        Preserve all strong V1 probabilities EXACTLY.

        Only:
            P(neutral) + P(sad) + P(fear)

        is redistributed among those same three classes according
        to the specialist adapter.
        """

        hybrid = base_probs.copy()

        weak_mass = float(
            sum(
                base_probs[index]
                for index in self.weak_indices
            )
        )

        for adapter_index, emotion in enumerate(
            self.weak_classes
        ):
            seven_index = self.class_to_index[
                emotion
            ]

            hybrid[
                seven_index
            ] = (
                weak_mass
                * adapter_probs[
                    adapter_index
                ]
            )

        # Sum should already be 1, but normalize for numerical safety.
        hybrid = self._normalize_probability_vector(
            hybrid
        )

        return (
            hybrid,
            weak_mass,
        )

    def _predict_temporal(self):
        sequence = np.stack(
            list(
                self.feature_buffer
            ),
            axis=0,
        ).astype(np.float32)

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_DIM
        ):
            raise ValueError(
                f"Temporal sequence shape error: {sequence.shape}"
            )

        normalized = (
            sequence
            - self.feature_mean
        ) / self.feature_std

        normalized = normalized[
            None,
            ...
        ]

        # ----------------------------------------------------
        # Original V1: 7 classes
        # ----------------------------------------------------

        base = self.temporal_model.predict(
            normalized,
            verbose=0,
        )[0]

        base = self._normalize_probability_vector(
            base
        )

        # ----------------------------------------------------
        # Specialist: only neutral / sad / fear
        # ----------------------------------------------------

        adapter = self.weak_adapter.predict(
            normalized,
            verbose=0,
        )[0]

        adapter = self._normalize_probability_vector(
            adapter
        )

        # ----------------------------------------------------
        # Hybrid probabilities
        # ----------------------------------------------------

        (
            hybrid_raw,
            weak_mass,
        ) = self._make_hybrid_probabilities(
            base,
            adapter,
        )

        self.base_probabilities = (
            base.copy()
        )

        self.adapter_probabilities = (
            adapter.copy()
        )

        self.hybrid_raw_probabilities = (
            hybrid_raw.copy()
        )

        self.weak_mass = (
            weak_mass
        )

        raw_top1, _, raw_conf, raw_margin = (
            self._top_stats(
                hybrid_raw
            )
        )

        raw_label = self.classes[
            raw_top1
        ]

        # ----------------------------------------------------
        # Same adaptive smoothing as the fixed realtime version
        # ----------------------------------------------------

        if self.smoothed_probabilities is None:
            smoothed = hybrid_raw.copy()

        else:
            current_label = (
                self.latest_label
            )

            strong_switch = (
                current_label is not None
                and raw_label != current_label
                and raw_conf
                >= SWITCH_MIN_CONFIDENCE
                and raw_margin
                >= SWITCH_MIN_MARGIN
            )

            if strong_switch:
                smoothed = (
                    hybrid_raw.copy()
                )

            else:
                alpha = (
                    self.smoothing_alpha
                )

                smoothed = (
                    alpha
                    * hybrid_raw
                    +
                    (
                        1.0
                        - alpha
                    )
                    * self.smoothed_probabilities
                )

        smoothed = (
            self._normalize_probability_vector(
                smoothed
            )
        )

        self.smoothed_probabilities = (
            smoothed
        )

        self.latest_probabilities = (
            smoothed.copy()
        )

        (
            best_index,
            _,
            confidence,
            margin,
        ) = self._top_stats(
            self.latest_probabilities
        )

        self.latest_label = (
            self.classes[
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
                margin,
            )
        )

    def update(
        self,
        face_bgr,
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
            self.latest_probabilities is None
            or
            self.samples_since_prediction
            >= self.predict_every_samples
        ):
            self._predict_temporal()
            self.samples_since_prediction = 0

        return self.get_state()

    def get_state(self):
        collected = len(
            self.feature_buffer
        )

        ready = (
            collected
            >= SEQUENCE_LENGTH
            and
            self.latest_probabilities
            is not None
        )

        final_probs = {}
        base_probs = {}
        adapter_probs = {}

        if self.latest_probabilities is not None:
            final_probs = {
                emotion:
                    float(
                        self.latest_probabilities[
                            index
                        ]
                    )
                for index, emotion in enumerate(
                    self.classes
                )
            }

        if self.base_probabilities is not None:
            base_probs = {
                emotion:
                    float(
                        self.base_probabilities[
                            index
                        ]
                    )
                for index, emotion in enumerate(
                    self.classes
                )
            }

        if self.adapter_probabilities is not None:
            adapter_probs = {
                emotion:
                    float(
                        self.adapter_probabilities[
                            index
                        ]
                    )
                for index, emotion in enumerate(
                    self.weak_classes
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
                final_probs,

            "base_probabilities":
                base_probs,

            "weak_adapter_probabilities":
                adapter_probs,

            "weak_mass":
                float(
                    self.weak_mass
                ),

            "weak_adapter_active":
                True,
        }


if __name__ == "__main__":
    detector = EmotionDetector()

    print(
        "All hybrid models loaded successfully."
    )

    print(
        f"Rolling Buffer: 0 / {SEQUENCE_LENGTH}"
    )

    print(
        "Ready for webcam integration."
    )
