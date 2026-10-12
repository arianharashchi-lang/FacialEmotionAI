import os
import json
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_ROOT = os.path.join(BASE_DIR, "data", "webcam_adaptation")
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "webcam_features")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_feature_extractor.keras")
METADATA_PATH = os.path.join(OUTPUT_ROOT, "metadata.json")

EMOTIONS = ["neutral", "sad", "fear"]
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 160
FEATURE_DIM = 256
BATCH_SIZE = 20


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found:\n{path}")


def get_person_folders():
    ensure_exists(INPUT_ROOT, "Webcam adaptation folder")
    persons = []
    for name in sorted(os.listdir(INPUT_ROOT)):
        full_path = os.path.join(INPUT_ROOT, name)
        if os.path.isdir(full_path):
            persons.append(name)
    return persons


def load_sequence(path):
    sequence = np.load(path)
    expected_shape = (SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)
    if sequence.shape != expected_shape:
        raise ValueError(
            f"Invalid sequence shape: {sequence.shape} "
            f"(expected {expected_shape})"
        )
    return sequence.astype(np.float32)


def extract_features(cnn_model, sequence):
    features = cnn_model.predict(
        sequence,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    features = np.asarray(features, dtype=np.float32)
    expected_shape = (SEQUENCE_LENGTH, FEATURE_DIM)
    if features.shape != expected_shape:
        raise ValueError(
            f"Unexpected CNN feature shape: {features.shape} "
            f"(expected {expected_shape})"
        )
    return features


def main():
    print()
    print("=" * 72)
    print("EXTRACT WEBCAM FEATURES FOR WEAK EMOTIONS")
    print("=" * 72)

    ensure_exists(MODEL_PATH, "CNN feature extractor")

    print()
    print("[INFO] Loading CNN feature extractor...")
    cnn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("[OK] CNN loaded")
    print("CNN output shape:", cnn_model.output_shape)

    persons = get_person_folders()
    if len(persons) == 0:
        raise RuntimeError(
            "No person folders found inside data/webcam_adaptation"
        )

    print()
    print("Persons found:", persons)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total_success = 0
    total_broken = 0

    metadata = {
        "persons": {},
        "total_success": 0,
        "total_broken": 0,
        "sequence_shape": [SEQUENCE_LENGTH, FEATURE_DIM],
        "color_pipeline": "BGR kept intentionally for V1 compatibility"
    }

    for person in persons:
        print()
        print("=" * 72)
        print(f"PERSON: {person}")
        print("=" * 72)

        metadata["persons"][person] = {}

        for emotion in EMOTIONS:
            input_dir = os.path.join(INPUT_ROOT, person, emotion)
            output_dir = os.path.join(OUTPUT_ROOT, person, emotion)
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.isdir(input_dir):
                print(f"[WARNING] Missing folder: {input_dir}")
                metadata["persons"][person][emotion] = {
                    "success": 0,
                    "broken": 0
                }
                continue

            files = sorted(
                filename
                for filename in os.listdir(input_dir)
                if filename.lower().endswith(".npy")
            )

            success_count = 0
            broken_count = 0

            for filename in files:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                try:
                    sequence = load_sequence(input_path)
                    features = extract_features(cnn_model, sequence)
                    np.save(output_path, features)
                    success_count += 1
                    total_success += 1
                except Exception as error:
                    broken_count += 1
                    total_broken += 1
                    print()
                    print("[BROKEN]", input_path)
                    print("Reason:", error)

            metadata["persons"][person][emotion] = {
                "success": success_count,
                "broken": broken_count
            }

            print(
                f"{emotion:10s} -> "
                f"success: {success_count:3d} | "
                f"broken: {broken_count:3d}"
            )

    metadata["total_success"] = total_success
    metadata["total_broken"] = total_broken

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print()
    print("=" * 72)
    print("WEBCAM FEATURE EXTRACTION COMPLETED")
    print("=" * 72)
    print("Total successful sequences:", total_success)
    print("Total broken sequences:", total_broken)
    print()
    print("Each saved sequence shape:")
    print(f"({SEQUENCE_LENGTH}, {FEATURE_DIM})")
    print()
    print("Saved to:")
    print(OUTPUT_ROOT)
    print()
    print("Metadata:")
    print(METADATA_PATH)


if __name__ == "__main__":
    main()
