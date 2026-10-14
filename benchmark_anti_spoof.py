import csv
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fake_detector import FakeDetector


BASE_DIR = Path(__file__).resolve().parent

DATA_ROOT = (
    BASE_DIR
    / "data"
    / "anti_spoof_eval"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "anti_spoofing_robustness"
)

CATEGORIES = [
    "real",
    "print",
    "phone",
    "monitor",
    "replay",
]

ATTACK_CATEGORIES = [
    "print",
    "phone",
    "monitor",
    "replay",
]

REAL_THRESHOLD = 0.65

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def iter_images(category):
    category_dir = (
        DATA_ROOT
        / category
    )

    if not category_dir.is_dir():
        return []

    return sorted(
        path
        for path in category_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in VALID_EXTENSIONS
        )
    )


def safe_rate(
    numerator,
    denominator,
):
    if denominator == 0:
        return None

    return float(
        numerator
        / denominator
    )


def save_confusion_matrix(
    matrix,
    output_path,
):
    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        "Anti-Spoofing Robustness Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    labels = [
        "Fake",
        "Real",
    ]

    plt.xticks(
        [0, 1],
        labels,
    )

    plt.yticks(
        [0, 1],
        labels,
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
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def save_category_plot(
    per_category,
    output_path,
):
    categories = []
    rejection_rates = []

    for category in ATTACK_CATEGORIES:
        metrics = per_category.get(
            category,
            {}
        )

        value = metrics.get(
            "spoof_rejection_rate"
        )

        if value is None:
            continue

        categories.append(
            category
        )

        rejection_rates.append(
            value * 100
        )

    if not categories:
        return

    plt.figure(
        figsize=(9, 5)
    )

    positions = np.arange(
        len(categories)
    )

    plt.bar(
        positions,
        rejection_rates,
    )

    plt.xticks(
        positions,
        categories,
    )

    plt.ylim(
        0,
        100,
    )

    plt.ylabel(
        "Spoof rejection rate (%)"
    )

    plt.title(
        "Anti-Spoofing Robustness by Attack Category"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = FakeDetector(
        real_threshold=REAL_THRESHOLD
    )

    if not detector.model_loaded:
        raise RuntimeError(
            "Could not load models/fake_discriminator.keras"
        )

    rows = []

    true_labels = []
    predicted_labels = []

    for category in CATEGORIES:
        paths = iter_images(
            category
        )

        if not paths:
            print(
                f"[WARNING] No images found for category: {category}"
            )
            continue

        expected_label = (
            1
            if category == "real"
            else 0
        )

        for path in paths:
            image = cv2.imread(
                str(path)
            )

            if image is None:
                print(
                    "[WARNING] Could not read:",
                    path,
                )
                continue

            result = detector.predict(
                image
            )

            if result["status"] == "MODEL_NOT_TRAINED":
                raise RuntimeError(
                    "Anti-spoofing model is not available."
                )

            predicted_label = (
                1
                if result["status"] == "REAL"
                else 0
            )

            relative_path = path.relative_to(
                BASE_DIR
            )

            rows.append(
                {
                    "path": str(
                        relative_path
                    ),
                    "category": category,
                    "expected": (
                        "REAL"
                        if expected_label == 1
                        else "FAKE"
                    ),
                    "predicted": result[
                        "status"
                    ],
                    "real_probability": float(
                        result[
                            "real_probability"
                        ]
                    ),
                    "fake_probability": float(
                        result[
                            "fake_probability"
                        ]
                    ),
                    "correct": bool(
                        predicted_label
                        == expected_label
                    ),
                }
            )

            true_labels.append(
                expected_label
            )

            predicted_labels.append(
                predicted_label
            )

    if not rows:
        raise RuntimeError(
            "No robustness evaluation images were found. "
            "Collect data under data/anti_spoof_eval first."
        )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision_real = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    recall_real = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    f1_real = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[
            0,
            1,
        ],
    )

    fake_total = int(
        sum(
            1
            for label in true_labels
            if label == 0
        )
    )

    fake_accepted_as_real = int(
        sum(
            1
            for true, pred
            in zip(
                true_labels,
                predicted_labels,
            )
            if (
                true == 0
                and pred == 1
            )
        )
    )

    real_total = int(
        sum(
            1
            for label in true_labels
            if label == 1
        )
    )

    real_rejected_as_fake = int(
        sum(
            1
            for true, pred
            in zip(
                true_labels,
                predicted_labels,
            )
            if (
                true == 1
                and pred == 0
            )
        )
    )

    apcer = safe_rate(
        fake_accepted_as_real,
        fake_total,
    )

    bpcer = safe_rate(
        real_rejected_as_fake,
        real_total,
    )

    acer = (
        None
        if (
            apcer is None
            or bpcer is None
        )
        else float(
            (
                apcer
                + bpcer
            )
            / 2.0
        )
    )

    per_category = {}

    for category in CATEGORIES:
        category_rows = [
            row
            for row in rows
            if row["category"]
            == category
        ]

        if not category_rows:
            continue

        count = len(
            category_rows
        )

        correct_count = sum(
            1
            for row in category_rows
            if row["correct"]
        )

        real_probabilities = [
            row["real_probability"]
            for row in category_rows
        ]

        category_metrics = {
            "count": count,
            "accuracy": float(
                correct_count
                / count
            ),
            "mean_real_probability": float(
                np.mean(
                    real_probabilities
                )
            ),
            "median_real_probability": float(
                np.median(
                    real_probabilities
                )
            ),
        }

        if category == "real":
            false_reject_count = sum(
                1
                for row in category_rows
                if row["predicted"]
                == "FAKE"
            )

            category_metrics[
                "false_reject_rate"
            ] = float(
                false_reject_count
                / count
            )

        else:
            accepted_as_real = sum(
                1
                for row in category_rows
                if row["predicted"]
                == "REAL"
            )

            category_metrics[
                "attack_error_rate"
            ] = float(
                accepted_as_real
                / count
            )

            category_metrics[
                "spoof_rejection_rate"
            ] = float(
                1.0
                - (
                    accepted_as_real
                    / count
                )
            )

        per_category[
            category
        ] = category_metrics

    metrics = {
        "protocol": {
            "real_threshold": (
                REAL_THRESHOLD
            ),
            "data_root": (
                "data/anti_spoof_eval"
            ),
            "categories": CATEGORIES,
            "note": (
                "This evaluation is intentionally separate from "
                "the random image validation split used during "
                "training. Results depend on the locally collected "
                "sessions and are not a universal PAD benchmark."
            ),
        },
        "overall": {
            "sample_count": len(
                rows
            ),
            "accuracy": float(
                accuracy
            ),
            "precision_real": float(
                precision_real
            ),
            "recall_real": float(
                recall_real
            ),
            "f1_real": float(
                f1_real
            ),
            "APCER": apcer,
            "BPCER": bpcer,
            "ACER": acer,
            "confusion_matrix": (
                matrix.tolist()
            ),
        },
        "per_category": (
            per_category
        ),
    }

    with (
        RESULT_DIR
        / "anti_spoof_robustness_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    with (
        RESULT_DIR
        / "predictions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "category",
                "expected",
                "predicted",
                "real_probability",
                "fake_probability",
                "correct",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    with (
        RESULT_DIR
        / "per_category.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "category",
            "count",
            "accuracy",
            "mean_real_probability",
            "median_real_probability",
            "attack_error_rate",
            "spoof_rejection_rate",
            "false_reject_rate",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for category in CATEGORIES:
            if category not in per_category:
                continue

            writer.writerow(
                {
                    "category": category,
                    **per_category[
                        category
                    ],
                }
            )

    save_confusion_matrix(
        matrix,
        RESULT_DIR
        / "anti_spoof_robustness_confusion_matrix.png",
    )

    save_category_plot(
        per_category,
        RESULT_DIR
        / "spoof_rejection_by_category.png",
    )

    print("")
    print("=" * 72)
    print("ANTI-SPOOFING ROBUSTNESS BENCHMARK")
    print("=" * 72)
    print(
        f"Samples : {len(rows)}"
    )
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    if apcer is not None:
        print(
            f"APCER   : {apcer:.4f}"
        )

    if bpcer is not None:
        print(
            f"BPCER   : {bpcer:.4f}"
        )

    if acer is not None:
        print(
            f"ACER    : {acer:.4f}"
        )

    print("")
    print("Per category")
    print("-" * 72)

    for category in CATEGORIES:
        values = per_category.get(
            category
        )

        if not values:
            print(
                f"{category:10s}: no samples"
            )
            continue

        line = (
            f"{category:10s}: "
            f"n={values['count']:4d} | "
            f"accuracy={values['accuracy']:.3f} | "
            f"mean real p="
            f"{values['mean_real_probability']:.3f}"
        )

        if category == "real":
            line += (
                " | false reject="
                f"{values['false_reject_rate']:.3f}"
            )
        else:
            line += (
                " | spoof rejection="
                f"{values['spoof_rejection_rate']:.3f}"
            )

        print(line)

    print("")
    print(
        "Results saved to:"
    )
    print(
        RESULT_DIR
    )
    print("")
    print(
        "Interpret APCER/BPCER only for this collected "
        "evaluation set; this is not a certified biometric "
        "presentation-attack benchmark."
    )


if __name__ == "__main__":
    main()
