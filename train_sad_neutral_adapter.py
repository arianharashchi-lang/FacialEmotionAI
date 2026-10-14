import argparse
import json
import os

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from emotion_logic import (
    temperature_scale_probs,
)


SEED = 42

np.random.seed(
    SEED
)

tf.random.set_seed(
    SEED
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "webcam_features",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "adapter",
)

MEAN_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_mean.npy",
)

STD_PATH = os.path.join(
    MODEL_DIR,
    "lstm_v1_feature_std.npy",
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sad_neutral_adapter.keras",
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "sad_neutral_adapter_classes.json",
)

CALIBRATION_PATH = os.path.join(
    MODEL_DIR,
    "sad_neutral_adapter_calibration.json",
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "sad_neutral_adapter_metrics.json",
)

REPORT_PATH = os.path.join(
    RESULT_DIR,
    "sad_neutral_adapter_cv_report.txt",
)


CLASSES = [
    "neutral",
    "sad",
]

SEQUENCE_LENGTH = 20
FEATURE_DIM = 256

BATCH_SIZE = 8
MAX_EPOCHS = 40
LEARNING_RATE = 2e-4

DEFAULT_MIN_PERSONS = 4
DEFAULT_MIN_SEQUENCES_PER_CLASS = 20
DEFAULT_TARGET_SELECTIVE_ACCURACY = 0.75
DEFAULT_MIN_COVERAGE = 0.25


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Sad/Neutral adapter using "
            "leave-one-person-out cross-validation, "
            "then calibrate confidence from out-of-fold predictions."
        )
    )

    parser.add_argument(
        "--min-persons",
        type=int,
        default=DEFAULT_MIN_PERSONS,
        help=(
            "Minimum number of participant folders required. "
            "Use 4 or more; 6 is recommended for this project."
        ),
    )

    parser.add_argument(
        "--min-sequences-per-class",
        type=int,
        default=DEFAULT_MIN_SEQUENCES_PER_CLASS,
        help=(
            "Minimum Neutral and Sad sequences required "
            "for every participant."
        ),
    )

    parser.add_argument(
        "--target-selective-accuracy",
        type=float,
        default=DEFAULT_TARGET_SELECTIVE_ACCURACY,
        help=(
            "Desired accuracy among samples where the adapter "
            "is allowed to act after calibration."
        ),
    )

    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=(
            "Minimum fraction of OOF samples that should remain "
            "eligible after confidence thresholding."
        ),
    )

    return parser.parse_args()


def ensure_exists(
    path,
    label,
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def get_persons():
    ensure_exists(
        DATA_ROOT,
        "Webcam features folder",
    )

    return [
        name
        for name in sorted(
            os.listdir(DATA_ROOT)
        )
        if os.path.isdir(
            os.path.join(
                DATA_ROOT,
                name,
            )
        )
    ]


def load_person(
    person_name,
):
    x_data = []
    y_data = []
    counts = {}

    person_dir = os.path.join(
        DATA_ROOT,
        person_name,
    )

    for class_index, emotion in enumerate(
        CLASSES
    ):
        emotion_dir = os.path.join(
            person_dir,
            emotion,
        )

        if not os.path.isdir(
            emotion_dir
        ):
            raise FileNotFoundError(
                f"Missing emotion folder:\n{emotion_dir}"
            )

        files = sorted(
            filename
            for filename in os.listdir(
                emotion_dir
            )
            if filename.lower().endswith(
                ".npy"
            )
        )

        counts[
            emotion
        ] = len(
            files
        )

        for filename in files:
            path = os.path.join(
                emotion_dir,
                filename,
            )

            sequence = np.load(
                path
            ).astype(
                np.float32
            )

            expected_shape = (
                SEQUENCE_LENGTH,
                FEATURE_DIM,
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
            dtype=np.float32,
        ),
        np.asarray(
            y_data,
            dtype=np.int32,
        ),
        counts,
    )


def normalize(
    x,
    feature_mean,
    feature_std,
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


def build_model():
    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_DIM,
        ),
        name="sad_neutral_sequence",
    )

    x = tf.keras.layers.GaussianNoise(
        0.008,
        name="feature_noise",
    )(
        inputs
    )

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            24,
            return_sequences=True,
            dropout=0.20,
        ),
        name="sad_neutral_bigru",
    )(
        x
    )

    x = tf.keras.layers.LayerNormalization(
        name="sad_neutral_norm",
    )(
        x
    )

    avg_pool = (
        tf.keras.layers.GlobalAveragePooling1D(
            name="avg_pool",
        )(
            x
        )
    )

    max_pool = (
        tf.keras.layers.GlobalMaxPooling1D(
            name="max_pool",
        )(
            x
        )
    )

    x = tf.keras.layers.Concatenate(
        name="pool_fusion",
    )(
        [
            avg_pool,
            max_pool,
        ]
    )

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=(
            tf.keras.regularizers.l2(
                2e-4
            )
        ),
        name="adapter_dense",
    )(
        x
    )

    x = tf.keras.layers.Dropout(
        0.35,
        name="adapter_dropout",
    )(
        x
    )

    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="sad_neutral_output",
    )(
        x
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="SadNeutralAdapter",
    )

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
        ],
    )

    return model


def negative_log_likelihood(
    labels,
    probabilities,
):
    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0,
    )

    selected = probabilities[
        np.arange(
            len(labels)
        ),
        labels,
    ]

    return float(
        -np.mean(
            np.log(
                selected
            )
        )
    )


def expected_calibration_error(
    labels,
    probabilities,
    bins=10,
):
    confidences = np.max(
        probabilities,
        axis=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    correct = (
        predictions
        == labels
    ).astype(
        np.float32
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for index in range(
        bins
    ):
        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if index == bins - 1:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if not np.any(
            mask
        ):
            continue

        bin_accuracy = float(
            np.mean(
                correct[
                    mask
                ]
            )
        )

        bin_confidence = float(
            np.mean(
                confidences[
                    mask
                ]
            )
        )

        ece += (
            float(
                np.mean(
                    mask
                )
            )
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(
        ece
    )


def fit_temperature(
    labels,
    probabilities,
):
    temperatures = np.linspace(
        0.50,
        3.00,
        251,
    )

    best_temperature = 1.0
    best_nll = None

    for temperature in temperatures:
        calibrated = np.asarray(
            [
                temperature_scale_probs(
                    row,
                    temperature,
                )
                for row in probabilities
            ],
            dtype=np.float32,
        )

        nll = negative_log_likelihood(
            labels,
            calibrated,
        )

        if (
            best_nll is None
            or nll < best_nll
        ):
            best_nll = nll
            best_temperature = float(
                temperature
            )

    return (
        best_temperature,
        float(
            best_nll
        ),
    )


def choose_confidence_threshold(
    labels,
    probabilities,
    target_accuracy,
    min_coverage,
):
    confidences = np.max(
        probabilities,
        axis=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    candidates = []

    for threshold in np.linspace(
        0.50,
        0.95,
        46,
    ):
        accepted = (
            confidences
            >= threshold
        )

        coverage = float(
            np.mean(
                accepted
            )
        )

        if (
            coverage
            < min_coverage
            or not np.any(
                accepted
            )
        ):
            continue

        selective_accuracy = float(
            accuracy_score(
                labels[
                    accepted
                ],
                predictions[
                    accepted
                ],
            )
        )

        selective_macro_f1 = float(
            f1_score(
                labels[
                    accepted
                ],
                predictions[
                    accepted
                ],
                average="macro",
                zero_division=0,
            )
        )

        candidates.append(
            {
                "threshold": float(
                    threshold
                ),
                "coverage": coverage,
                "selective_accuracy": (
                    selective_accuracy
                ),
                "selective_macro_f1": (
                    selective_macro_f1
                ),
            }
        )

    if not candidates:
        return {
            "threshold": 0.70,
            "coverage": 0.0,
            "selective_accuracy": 0.0,
            "selective_macro_f1": 0.0,
            "selection_rule": (
                "fallback_no_candidate"
            ),
        }

    target_candidates = [
        item
        for item in candidates
        if item[
            "selective_accuracy"
        ] >= target_accuracy
    ]

    if target_candidates:
        selected = max(
            target_candidates,
            key=lambda item: (
                item["coverage"],
                item[
                    "selective_macro_f1"
                ],
            ),
        ).copy()

        selected[
            "selection_rule"
        ] = (
            "max_coverage_meeting_target_accuracy"
        )

        return selected

    selected = max(
        candidates,
        key=lambda item: (
            item[
                "selective_accuracy"
            ]
            * item[
                "coverage"
            ],
            item[
                "selective_macro_f1"
            ],
        ),
    ).copy()

    selected[
        "selection_rule"
    ] = (
        "fallback_best_accuracy_coverage_product"
    )

    return selected


def main():
    args = parse_args()

    if args.min_persons < 3:
        raise ValueError(
            "--min-persons must be at least 3."
        )

    if (
        not 0.50
        <= args.target_selective_accuracy
        <= 1.0
    ):
        raise ValueError(
            "--target-selective-accuracy "
            "must be between 0.50 and 1.0."
        )

    if (
        not 0.05
        <= args.min_coverage
        <= 1.0
    ):
        raise ValueError(
            "--min-coverage must be "
            "between 0.05 and 1.0."
        )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    ensure_exists(
        MEAN_PATH,
        "V1 feature mean",
    )

    ensure_exists(
        STD_PATH,
        "V1 feature std",
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
        feature_std,
    )

    persons = get_persons()

    if len(persons) < args.min_persons:
        raise RuntimeError(
            f"Found {len(persons)} participants, "
            f"but at least {args.min_persons} are required. "
            "Collect more people before retraining."
        )

    print("")
    print("=" * 72)
    print("EXPANDED SAD / NEUTRAL ADAPTER")
    print("=" * 72)
    print(
        "Participants found:",
        persons,
    )

    person_data = {}
    participant_counts = {}

    for person in persons:
        (
            x_person,
            y_person,
            counts,
        ) = load_person(
            person
        )

        missing = [
            emotion
            for emotion in CLASSES
            if counts.get(
                emotion,
                0,
            )
            < args.min_sequences_per_class
        ]

        if missing:
            raise RuntimeError(
                f"{person} does not meet the minimum "
                f"{args.min_sequences_per_class} sequences "
                f"for: {missing}. Counts: {counts}"
            )

        x_person = normalize(
            x_person,
            feature_mean,
            feature_std,
        )

        person_data[
            person
        ] = (
            x_person,
            y_person,
        )

        participant_counts[
            person
        ] = counts

        print(
            f"{person}: "
            f"neutral={counts['neutral']} | "
            f"sad={counts['sad']}"
        )

    fold_results = []
    best_epochs = []
    report_lines = []

    oof_probabilities = []
    oof_labels = []
    oof_persons = []

    print("")
    print("=" * 72)
    print("LEAVE-ONE-PERSON-OUT CROSS-VALIDATION")
    print("=" * 72)

    for fold_index, val_person in enumerate(
        persons,
        start=1,
    ):
        train_persons = [
            person
            for person in persons
            if person != val_person
        ]

        x_train = np.concatenate(
            [
                person_data[
                    person
                ][0]
                for person in train_persons
            ],
            axis=0,
        )

        y_train = np.concatenate(
            [
                person_data[
                    person
                ][1]
                for person in train_persons
            ],
            axis=0,
        )

        (
            x_val,
            y_val,
        ) = person_data[
            val_person
        ]

        tf.keras.backend.clear_session()

        model = build_model()

        early_stopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=7,
                restore_best_weights=True,
                verbose=0,
            )
        )

        reduce_lr = (
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0,
            )
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(
                x_val,
                y_val,
            ),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[
                early_stopping,
                reduce_lr,
            ],
            verbose=0,
        )

        best_epoch = int(
            np.argmin(
                history.history[
                    "val_loss"
                ]
            )
            + 1
        )

        best_epochs.append(
            best_epoch
        )

        probabilities = model.predict(
            x_val,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        accuracy = float(
            accuracy_score(
                y_val,
                predictions,
            )
        )

        macro_f1 = float(
            f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        per_class_f1 = (
            f1_score(
                y_val,
                predictions,
                average=None,
                labels=[
                    0,
                    1,
                ],
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_val,
            predictions,
            labels=[
                0,
                1,
            ],
        )

        fold_results.append(
            {
                "validation_person": (
                    val_person
                ),
                "train_persons": (
                    train_persons
                ),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "neutral_f1": float(
                    per_class_f1[
                        0
                    ]
                ),
                "sad_f1": float(
                    per_class_f1[
                        1
                    ]
                ),
                "best_epoch": (
                    best_epoch
                ),
                "confusion_matrix": (
                    matrix.tolist()
                ),
            }
        )

        oof_probabilities.append(
            probabilities
        )

        oof_labels.append(
            y_val.copy()
        )

        oof_persons.extend(
            [
                val_person
            ]
            * len(
                y_val
            )
        )

        report = classification_report(
            y_val,
            predictions,
            target_names=CLASSES,
            zero_division=0,
        )

        report_lines.append(
            "\n"
            + "=" * 72
            + "\n"
            + (
                f"VALIDATION PERSON: "
                f"{val_person}\n"
            )
            + "=" * 72
            + "\n"
            + report
            + "\n"
        )

        print(
            f"{val_person}: "
            f"accuracy={accuracy:.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"neutral_f1={per_class_f1[0]:.4f} | "
            f"sad_f1={per_class_f1[1]:.4f}"
        )

    oof_probabilities = np.concatenate(
        oof_probabilities,
        axis=0,
    )

    oof_labels = np.concatenate(
        oof_labels,
        axis=0,
    )

    raw_oof_predictions = np.argmax(
        oof_probabilities,
        axis=1,
    )

    raw_oof_accuracy = float(
        accuracy_score(
            oof_labels,
            raw_oof_predictions,
        )
    )

    raw_oof_macro_f1 = float(
        f1_score(
            oof_labels,
            raw_oof_predictions,
            average="macro",
            zero_division=0,
        )
    )

    raw_ece = (
        expected_calibration_error(
            oof_labels,
            oof_probabilities,
        )
    )

    raw_nll = (
        negative_log_likelihood(
            oof_labels,
            oof_probabilities,
        )
    )

    (
        temperature,
        calibrated_nll,
    ) = fit_temperature(
        oof_labels,
        oof_probabilities,
    )

    calibrated_oof = np.asarray(
        [
            temperature_scale_probs(
                row,
                temperature,
            )
            for row in oof_probabilities
        ],
        dtype=np.float32,
    )

    calibrated_ece = (
        expected_calibration_error(
            oof_labels,
            calibrated_oof,
        )
    )

    threshold_result = (
        choose_confidence_threshold(
            labels=oof_labels,
            probabilities=calibrated_oof,
            target_accuracy=(
                args.target_selective_accuracy
            ),
            min_coverage=(
                args.min_coverage
            ),
        )
    )

    mean_accuracy = float(
        np.mean(
            [
                result[
                    "accuracy"
                ]
                for result in fold_results
            ]
        )
    )

    mean_macro_f1 = float(
        np.mean(
            [
                result[
                    "macro_f1"
                ]
                for result in fold_results
            ]
        )
    )

    mean_neutral_f1 = float(
        np.mean(
            [
                result[
                    "neutral_f1"
                ]
                for result in fold_results
            ]
        )
    )

    mean_sad_f1 = float(
        np.mean(
            [
                result[
                    "sad_f1"
                ]
                for result in fold_results
            ]
        )
    )

    final_epochs = max(
        3,
        int(
            round(
                np.mean(
                    best_epochs
                )
            )
        ),
    )

    x_all = np.concatenate(
        [
            person_data[
                person
            ][0]
            for person in persons
        ],
        axis=0,
    )

    y_all = np.concatenate(
        [
            person_data[
                person
            ][1]
            for person in persons
        ],
        axis=0,
    )

    tf.keras.backend.clear_session()

    final_model = build_model()

    print("")
    print("=" * 72)
    print("TRAINING FINAL EXPANDED ADAPTER")
    print("=" * 72)
    print(
        "Participants:",
        len(
            persons
        ),
    )
    print(
        "Total sequences:",
        len(
            y_all
        ),
    )
    print(
        "Final epochs:",
        final_epochs,
    )

    final_model.fit(
        x_all,
        y_all,
        epochs=final_epochs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1,
    )

    final_model.save(
        FINAL_MODEL_PATH
    )

    with open(
        CLASS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            CLASSES,
            file,
            indent=4,
        )

    calibration = {
        "version": 1,
        "temperature": float(
            temperature
        ),
        "confidence_threshold": float(
            threshold_result[
                "threshold"
            ]
        ),
        "minimum_group_mass": 0.15,
        "calibration_source": (
            "leave-one-person-out out-of-fold predictions"
        ),
        "participant_count": len(
            persons
        ),
        "oof_sample_count": int(
            len(
                oof_labels
            )
        ),
        "raw_nll": raw_nll,
        "calibrated_nll": calibrated_nll,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "target_selective_accuracy": (
            args.target_selective_accuracy
        ),
        "minimum_coverage": (
            args.min_coverage
        ),
        "selected_threshold_metrics": (
            threshold_result
        ),
    }

    with open(
        CALIBRATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            calibration,
            file,
            indent=4,
        )

    metrics = {
        "classes": CLASSES,
        "participant_count": len(
            persons
        ),
        "persons": persons,
        "participant_counts": (
            participant_counts
        ),
        "total_sequences": int(
            len(
                y_all
            )
        ),
        "folds": fold_results,
        "mean_accuracy": (
            mean_accuracy
        ),
        "mean_macro_f1": (
            mean_macro_f1
        ),
        "mean_neutral_f1": (
            mean_neutral_f1
        ),
        "mean_sad_f1": (
            mean_sad_f1
        ),
        "oof_accuracy": (
            raw_oof_accuracy
        ),
        "oof_macro_f1": (
            raw_oof_macro_f1
        ),
        "final_training_epochs": (
            final_epochs
        ),
        "calibration": calibration,
        "important_note": (
            "The adapter is trained only on neutral and sad "
            "webcam features. Person-wise validation is used, "
            "then temperature scaling and the activation threshold "
            "are estimated only from out-of-fold predictions. "
            "The strong V1 model weights are not modified."
        ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                report_lines
            )
        )

        file.write(
            "\n"
            + "=" * 72
            + "\nEXPANDED ADAPTER SUMMARY\n"
            + "=" * 72
            + "\n"
        )

        file.write(
            f"Participants: {len(persons)}\n"
        )

        file.write(
            f"Total sequences: {len(y_all)}\n"
        )

        file.write(
            f"Mean Accuracy: {mean_accuracy:.6f}\n"
        )

        file.write(
            f"Mean Macro F1: {mean_macro_f1:.6f}\n"
        )

        file.write(
            f"Mean Neutral F1: {mean_neutral_f1:.6f}\n"
        )

        file.write(
            f"Mean Sad F1: {mean_sad_f1:.6f}\n"
        )

        file.write(
            f"OOF Accuracy: {raw_oof_accuracy:.6f}\n"
        )

        file.write(
            f"OOF Macro F1: {raw_oof_macro_f1:.6f}\n"
        )

        file.write(
            f"Temperature: {temperature:.6f}\n"
        )

        file.write(
            f"Raw ECE: {raw_ece:.6f}\n"
        )

        file.write(
            f"Calibrated ECE: {calibrated_ece:.6f}\n"
        )

        file.write(
            "Confidence threshold: "
            f"{threshold_result['threshold']:.6f}\n"
        )

        file.write(
            "Selective accuracy: "
            f"{threshold_result['selective_accuracy']:.6f}\n"
        )

        file.write(
            "Coverage: "
            f"{threshold_result['coverage']:.6f}\n"
        )

        file.write(
            f"Final epochs: {final_epochs}\n"
        )

    print("")
    print("=" * 72)
    print("EXPANDED SAD / NEUTRAL ADAPTER READY")
    print("=" * 72)
    print(
        f"Mean Accuracy   : {mean_accuracy:.4f}"
    )
    print(
        f"Mean Macro F1   : {mean_macro_f1:.4f}"
    )
    print(
        f"Mean Neutral F1 : {mean_neutral_f1:.4f}"
    )
    print(
        f"Mean Sad F1     : {mean_sad_f1:.4f}"
    )
    print(
        f"Temperature     : {temperature:.3f}"
    )
    print(
        f"ECE raw -> calibrated: "
        f"{raw_ece:.4f} -> {calibrated_ece:.4f}"
    )
    print(
        "Confidence threshold: "
        f"{threshold_result['threshold']:.2f}"
    )
    print(
        "Selective accuracy: "
        f"{threshold_result['selective_accuracy']:.3f}"
    )
    print(
        "Coverage: "
        f"{threshold_result['coverage']:.3f}"
    )
    print(
        "Model saved:",
        FINAL_MODEL_PATH,
    )
    print(
        "Calibration saved:",
        CALIBRATION_PATH,
    )


if __name__ == "__main__":
    main()
