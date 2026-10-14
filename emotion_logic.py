"""Pure inference helpers for the deployed five-class emotion pipeline.

This module intentionally has no TensorFlow, OpenCV, webcam, dataset, or model-file
dependencies so the core probability logic can be unit-tested quickly in CI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def normalize_probs(probabilities: Sequence[float]) -> np.ndarray:
    """Return a finite, non-negative probability vector that sums to one."""
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
        np.sum(
            probabilities
        )
    )

    if total <= 0:
        return np.full(
            len(probabilities),
            1.0 / len(probabilities),
            dtype=np.float32,
        )

    return probabilities / total


def top_stats(probabilities: Sequence[float]) -> tuple[int, int, float, float]:
    """Return top-1 index, top-2 index, confidence, and top-2 margin."""
    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

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


def reliability(confidence: float, margin: float) -> str:
    """Map confidence and top-2 margin to the runtime reliability label."""
    if (
        confidence >= 0.60
        and margin >= 0.15
    ):
        return "HIGH"

    if (
        confidence >= 0.45
        and margin >= 0.08
    ):
        return "MEDIUM"

    return "LOW"


def merge_v1_to_5(
    base_7: Sequence[float],
    v1_index: Mapping[str, int],
) -> np.ndarray:
    """Fuse the original seven classes into the deployed five classes."""
    angry = base_7[
        v1_index["angry"]
    ]

    happy = base_7[
        v1_index["happy"]
    ]

    surprise = (
        base_7[
            v1_index["surprise"]
        ]
        + base_7[
            v1_index["fear"]
        ]
    )

    sad = (
        base_7[
            v1_index["sad"]
        ]
        + base_7[
            v1_index["disgust"]
        ]
    )

    neutral = base_7[
        v1_index["neutral"]
    ]

    merged = np.asarray(
        [
            angry,
            happy,
            surprise,
            sad,
            neutral,
        ],
        dtype=np.float32,
    )

    return normalize_probs(
        merged
    )


def apply_adapter_if_confident(
    merged_5: Sequence[float],
    adapter_probs: Sequence[float],
    adapter_classes: Sequence[str],
    confidence_threshold: float,
    min_group_mass: float,
    final_classes: Sequence[str],
) -> tuple[np.ndarray, bool, float, float]:
    """Redistribute only Sad/Neutral mass when the adapter is trusted."""
    final_probs = np.asarray(
        merged_5,
        dtype=np.float32,
    ).copy()

    adapter_probs = np.asarray(
        adapter_probs,
        dtype=np.float32,
    )

    sad_index = final_classes.index(
        "sad"
    )

    neutral_index = final_classes.index(
        "neutral"
    )

    group_mass = float(
        final_probs[sad_index]
        + final_probs[neutral_index]
    )

    adapter_top = float(
        np.max(
            adapter_probs
        )
    )

    use_adapter = (
        adapter_top
        >= confidence_threshold
        and group_mass
        >= min_group_mass
    )

    if not use_adapter:
        return (
            final_probs,
            False,
            group_mass,
            adapter_top,
        )

    adapter_map = {
        emotion: float(
            adapter_probs[index]
        )
        for index, emotion in enumerate(
            adapter_classes
        )
    }

    final_probs[
        neutral_index
    ] = (
        group_mass
        * adapter_map["neutral"]
    )

    final_probs[
        sad_index
    ] = (
        group_mass
        * adapter_map["sad"]
    )

    return (
        normalize_probs(
            final_probs
        ),
        True,
        group_mass,
        adapter_top,
    )
