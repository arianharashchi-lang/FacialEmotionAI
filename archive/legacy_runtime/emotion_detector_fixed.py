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

IMAGE_SIZE = 160
SEQUENCE_LENGTH = 20
FEATURE_DIM = 256

# 20 samples cover about 2.4 seconds. This is closer to the temporal span
# used by the RAVDESS sequences than feeding every webcam frame.
SAMPLE_INTERVAL = 0.12

# Update on every new sampled frame after the buffer is full.
PREDICT_EVERY_SAMPLES = 1

# Higher than the previous 0.35 so the live result adapts faster.
SMOOTHING_ALPHA = 0.70

# If the raw model clearly changes class, do not let old smoothed
# probabilities keep the previous emotion on screen for too long.
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


@tf.keras.utils.register_keras_serializable(package="EmotionAI")
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # These names must match the V1 Rebuild training class.
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
    def __init__(
        self,
        sample_interval=SAMPLE_INTERVAL,
        predict_every_samples=PREDICT_EVERY_SAMPLES,
        smoothing_alpha=SMOOTHING_ALPHA,
    ):
        print("\n" + "=" * 70)
        print("LOADING EMOTION DETECTOR - FIXED REALTIME")
        print("=" * 70)

        self.sample_interval = float(sample_interval)
        self.predict_every_samples = int(predict_every_samples)
        self.smoothing_alpha = float(smoothing_alpha)

        self._check_files()
        self.classes = self._load_classes()

        self.feature_mean = np.load(MEAN_PATH).astype(np.float32)
        self.feature_std = np.load(STD_PATH).astype(np.float32)
        self.feature_std = np.where(self.feature_std < 1e-6, 1e-6, self.feature_std)

        print("[INFO] Loading CNN feature extractor...")
        self.cnn_model = tf.keras.models.load_model(
            CNN_FEATURE_MODEL_PATH,
            compile=False,
        )
        print("[OK] CNN feature extractor loaded:", self.cnn_model.output_shape)

        print("[INFO] Loading LSTM V1 Rebuild...")
        custom_objects = {
            "TemporalAttention": TemporalAttention,
            "EmotionAI>TemporalAttention": TemporalAttention,
        }
        self.temporal_model = tf.keras.models.load_model(
            TEMPORAL_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
        )
        print("[OK] LSTM V1 loaded:", self.temporal_model.input_shape, "->", self.temporal_model.output_shape)

        if int(self.temporal_model.output_shape[-1]) != len(self.classes):
            raise ValueError(
                f"Class count mismatch: model={self.temporal_model.output_shape[-1]}, classes={len(self.classes)}"
            )

        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.last_sample_time = 0.0
        self.samples_since_prediction = 0
        self.latest_probabilities = None
        self.raw_probabilities = None
        self.smoothed_probabilities = None
        self.latest_label = None
        self.latest_confidence = 0.0
        self.latest_margin = 0.0
        self.latest_reliability = "WARMUP"

        print("=" * 70)
        print("EMOTION DETECTOR READY")
        print("=" * 70 + "\n")

    def _check_files(self):
        for path in [
            CNN_FEATURE_MODEL_PATH,
            TEMPORAL_MODEL_PATH,
            MEAN_PATH,
            STD_PATH,
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required file not found:\n{path}")

    def _load_classes(self):
        if not os.path.isfile(CLASS_PATH):
            return DEFAULT_CLASSES.copy()
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "classes" in data:
                return data["classes"]
        except Exception as exc:
            print("[WARNING] Could not load class file:", exc)
        return DEFAULT_CLASSES.copy()

    def reset(self):
        """Hard reset for a new person/test/expression session."""
        self.feature_buffer.clear()
        self.last_sample_time = 0.0
        self.samples_since_prediction = 0
        self.latest_probabilities = None
        self.raw_probabilities = None
        self.smoothed_probabilities = None
        self.latest_label = None
        self.latest_confidence = 0.0
        self.latest_margin = 0.0
        self.latest_reliability = "WARMUP"

    def reset_probability_history(self):
        """Keep the 20-frame buffer but remove old probability inertia."""
        self.latest_probabilities = None
        self.raw_probabilities = None
        self.smoothed_probabilities = None
        self.latest_label = None
        self.latest_confidence = 0.0
        self.latest_margin = 0.0
        self.latest_reliability = "WARMUP"
        self.samples_since_prediction = self.predict_every_samples

    def _extract_feature(self, face_bgr):
        resized = cv2.resize(
            face_bgr,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # IMPORTANT: V1 temporal features were extracted from OpenCV BGR
        # sequences. Keep BGR here to match that temporal training pipeline.
        image = np.clip(resized, 0, 255).astype(np.float32)[None, ...]

        feature = self.cnn_model.predict(image, verbose=0)[0]
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)

        if feature.shape[0] != FEATURE_DIM:
            raise ValueError(
                f"CNN feature dimension mismatch: expected {FEATURE_DIM}, got {feature.shape}"
            )

        return feature

    @staticmethod
    def _top_stats(probabilities):
        order = np.argsort(probabilities)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        confidence = float(probabilities[top1])
        margin = float(probabilities[top1] - probabilities[top2])
        return top1, top2, confidence, margin

    @staticmethod
    def _reliability(confidence, margin):
        if confidence >= 0.60 and margin >= 0.15:
            return "HIGH"
        if confidence >= 0.45 and margin >= 0.08:
            return "MEDIUM"
        return "LOW"

    def _predict_temporal(self):
        sequence = np.stack(list(self.feature_buffer), axis=0).astype(np.float32)
        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            raise ValueError(f"Temporal sequence shape error: {sequence.shape}")

        sequence = (sequence - self.feature_mean) / self.feature_std
        sequence = sequence[None, ...]

        raw = self.temporal_model.predict(sequence, verbose=0)[0]
        raw = np.asarray(raw, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.maximum(raw, 0.0)
        raw_sum = float(raw.sum())
        if raw_sum <= 0:
            raw = np.full(len(self.classes), 1.0 / len(self.classes), dtype=np.float32)
        else:
            raw /= raw_sum

        self.raw_probabilities = raw.copy()
        raw_top1, _, raw_conf, raw_margin = self._top_stats(raw)
        raw_label = self.classes[raw_top1]

        # Adaptive smoothing: if the raw network has clearly moved to a new
        # class, immediately release the old history. This fixes the common
        # "I changed to neutral but the old happy/angry result stays" issue.
        if self.smoothed_probabilities is None:
            smoothed = raw.copy()
        else:
            current_label = self.latest_label
            strong_switch = (
                current_label is not None
                and raw_label != current_label
                and raw_conf >= SWITCH_MIN_CONFIDENCE
                and raw_margin >= SWITCH_MIN_MARGIN
            )

            if strong_switch:
                smoothed = raw.copy()
            else:
                alpha = self.smoothing_alpha
                smoothed = alpha * raw + (1.0 - alpha) * self.smoothed_probabilities

        smoothed = np.maximum(smoothed, 0.0)
        total = float(smoothed.sum())
        if total > 0:
            smoothed /= total

        self.smoothed_probabilities = smoothed
        self.latest_probabilities = smoothed.copy()

        best_index, _, confidence, margin = self._top_stats(self.latest_probabilities)
        self.latest_label = self.classes[best_index]
        self.latest_confidence = confidence
        self.latest_margin = margin
        self.latest_reliability = self._reliability(confidence, margin)

    def update(self, face_bgr):
        current_time = time.monotonic()
        if current_time - self.last_sample_time < self.sample_interval:
            return self.get_state()

        self.last_sample_time = current_time
        feature = self._extract_feature(face_bgr)
        self.feature_buffer.append(feature)
        self.samples_since_prediction += 1

        if len(self.feature_buffer) < SEQUENCE_LENGTH:
            return self.get_state()

        if (
            self.latest_probabilities is None
            or self.samples_since_prediction >= self.predict_every_samples
        ):
            self._predict_temporal()
            self.samples_since_prediction = 0

        return self.get_state()

    def get_state(self):
        collected = len(self.feature_buffer)
        ready = collected >= SEQUENCE_LENGTH and self.latest_probabilities is not None

        probs = {}
        raw_probs = {}
        if self.latest_probabilities is not None:
            probs = {
                emotion: float(self.latest_probabilities[i])
                for i, emotion in enumerate(self.classes)
            }
        if self.raw_probabilities is not None:
            raw_probs = {
                emotion: float(self.raw_probabilities[i])
                for i, emotion in enumerate(self.classes)
            }

        return {
            "ready": ready,
            "collected": collected,
            "required": SEQUENCE_LENGTH,
            "label": self.latest_label,
            "confidence": self.latest_confidence,
            "margin": self.latest_margin,
            "reliability": self.latest_reliability,
            "probabilities": probs,
            "raw_probabilities": raw_probs,
        }


if __name__ == "__main__":
    detector = EmotionDetector()
    print("All final emotion models loaded successfully.")
    print(f"Rolling Buffer: 0 / {SEQUENCE_LENGTH}")
    print("Ready for webcam integration.")
