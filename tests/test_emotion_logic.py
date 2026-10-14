import numpy as np

from emotion_logic import (
    apply_adapter_if_confident,
    merge_v1_to_5,
    normalize_probs,
    reliability,
    top_stats,
)


V1_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

V1_INDEX = {
    name: index
    for index, name in enumerate(V1_CLASSES)
}

FINAL_CLASSES = [
    "angry",
    "happy",
    "surprise",
    "sad",
    "neutral",
]


def test_normalize_probs_sanitizes_invalid_values():
    result = normalize_probs(
        [np.nan, -3.0, 2.0, np.inf]
    )

    np.testing.assert_allclose(
        result,
        [0.0, 0.0, 1.0, 0.0],
    )
    assert np.isclose(
        result.sum(),
        1.0,
    )


def test_normalize_probs_uses_uniform_distribution_for_zero_mass():
    result = normalize_probs(
        [0.0, 0.0, 0.0, 0.0]
    )

    np.testing.assert_allclose(
        result,
        [0.25, 0.25, 0.25, 0.25],
    )


def test_merge_v1_to_5_fuses_fear_and_disgust_into_runtime_classes():
    base_7 = np.asarray(
        [
            0.10,  # angry
            0.05,  # disgust
            0.15,  # fear
            0.20,  # happy
            0.25,  # neutral
            0.10,  # sad
            0.15,  # surprise
        ],
        dtype=np.float32,
    )

    result = merge_v1_to_5(
        base_7,
        V1_INDEX,
    )

    np.testing.assert_allclose(
        result,
        [
            0.10,  # angry
            0.20,  # happy
            0.30,  # surprise + fear
            0.15,  # sad + disgust
            0.25,  # neutral
        ],
        atol=1e-6,
    )
    assert np.isclose(
        result.sum(),
        1.0,
    )


def test_adapter_redistributes_only_sad_neutral_mass_when_confident():
    merged = np.asarray(
        [0.10, 0.20, 0.20, 0.30, 0.20],
        dtype=np.float32,
    )

    result, active, group_mass, adapter_top = (
        apply_adapter_if_confident(
            merged_5=merged,
            adapter_probs=[0.80, 0.20],
            adapter_classes=["neutral", "sad"],
            confidence_threshold=0.70,
            min_group_mass=0.15,
            final_classes=FINAL_CLASSES,
        )
    )

    assert active is True
    assert np.isclose(
        group_mass,
        0.50,
    )
    assert np.isclose(
        adapter_top,
        0.80,
    )

    np.testing.assert_allclose(
        result[:3],
        merged[:3],
        atol=1e-6,
    )
    assert np.isclose(
        result[3],
        0.10,
    )
    assert np.isclose(
        result[4],
        0.40,
    )
    assert np.isclose(
        result.sum(),
        1.0,
    )


def test_adapter_bypasses_when_confidence_is_below_threshold():
    merged = np.asarray(
        [0.10, 0.20, 0.20, 0.30, 0.20],
        dtype=np.float32,
    )

    result, active, _, adapter_top = (
        apply_adapter_if_confident(
            merged_5=merged,
            adapter_probs=[0.60, 0.40],
            adapter_classes=["neutral", "sad"],
            confidence_threshold=0.70,
            min_group_mass=0.15,
            final_classes=FINAL_CLASSES,
        )
    )

    assert active is False
    assert np.isclose(
        adapter_top,
        0.60,
    )
    np.testing.assert_allclose(
        result,
        merged,
        atol=1e-6,
    )


def test_adapter_bypasses_when_sad_neutral_mass_is_too_small():
    merged = np.asarray(
        [0.35, 0.35, 0.20, 0.05, 0.05],
        dtype=np.float32,
    )

    result, active, group_mass, _ = (
        apply_adapter_if_confident(
            merged_5=merged,
            adapter_probs=[0.10, 0.90],
            adapter_classes=["neutral", "sad"],
            confidence_threshold=0.70,
            min_group_mass=0.15,
            final_classes=FINAL_CLASSES,
        )
    )

    assert active is False
    assert np.isclose(
        group_mass,
        0.10,
    )
    np.testing.assert_allclose(
        result,
        merged,
        atol=1e-6,
    )


def test_adapter_uses_class_names_instead_of_assuming_order():
    merged = np.asarray(
        [0.10, 0.20, 0.20, 0.30, 0.20],
        dtype=np.float32,
    )

    result, active, _, _ = (
        apply_adapter_if_confident(
            merged_5=merged,
            adapter_probs=[0.90, 0.10],
            adapter_classes=["sad", "neutral"],
            confidence_threshold=0.70,
            min_group_mass=0.15,
            final_classes=FINAL_CLASSES,
        )
    )

    assert active is True
    assert np.isclose(
        result[3],
        0.45,
    )
    assert np.isclose(
        result[4],
        0.05,
    )


def test_top_stats_returns_expected_confidence_and_margin():
    top1, top2, confidence, margin = top_stats(
        [0.10, 0.55, 0.25, 0.10]
    )

    assert top1 == 1
    assert top2 == 2
    assert np.isclose(
        confidence,
        0.55,
    )
    assert np.isclose(
        margin,
        0.30,
    )


def test_reliability_boundaries_match_runtime_policy():
    assert reliability(
        0.60,
        0.15,
    ) == "HIGH"

    assert reliability(
        0.45,
        0.08,
    ) == "MEDIUM"

    assert reliability(
        0.44,
        0.20,
    ) == "LOW"

    assert reliability(
        0.80,
        0.07,
    ) == "LOW"
