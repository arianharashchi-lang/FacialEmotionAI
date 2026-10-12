import os
import shutil
import hashlib
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RAW_DATASET_DIR = os.path.join(
    DATA_DIR,
    "raw_emotion_dataset"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "emotion_rgb"
)


# ============================================================
# FINAL CLASSES
# ============================================================

CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ============================================================
# DIFFERENT POSSIBLE FOLDER NAMES
# ============================================================

CLASS_ALIASES = {

    "angry": {
        "angry",
        "anger",
        "angriness"
    },

    "disgust": {
        "disgust",
        "disgusted"
    },

    "fear": {
        "fear",
        "fearful",
        "scared"
    },

    "happy": {
        "happy",
        "happiness",
        "joy"
    },

    "neutral": {
        "neutral",
        "normal"
    },

    "sad": {
        "sad",
        "sadness"
    },

    "surprise": {
        "surprise",
        "surprised"
    }
}


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for emotion in CLASSES:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                emotion
            ),
            exist_ok=True
        )


# ============================================================
# DETECT CLASS FROM PATH
# ============================================================

def detect_emotion(path):

    path_parts = [
        part.lower().strip()
        for part in os.path.normpath(path).split(os.sep)
    ]

    for emotion, aliases in CLASS_ALIASES.items():

        for part in path_parts:

            if part in aliases:

                return emotion

    return None


# ============================================================
# CHECK IMAGE
# ============================================================

def valid_image(image_path):

    try:

        with Image.open(image_path) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# HASH IMAGE
# ============================================================

def file_hash(filepath):

    sha = hashlib.sha1()

    with open(filepath, "rb") as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset():

    print("")
    print("=" * 70)
    print("EMOTION RGB DATASET PREPARATION")
    print("=" * 70)

    if not os.path.isdir(
        RAW_DATASET_DIR
    ):

        print("")
        print("[ERROR] Raw dataset folder not found:")
        print(RAW_DATASET_DIR)

        print("")
        print(
            "Create data/raw_emotion_dataset "
            "and put the downloaded dataset inside it."
        )

        return


    create_output_folders()


    counters = {
        emotion: 0
        for emotion in CLASSES
    }

    bad_images = 0

    ignored_images = 0

    duplicated_images = 0

    hashes = set()


    # ========================================================
    # SEARCH EVERYTHING RECURSIVELY
    # ========================================================

    for root, dirs, files in os.walk(
        RAW_DATASET_DIR
    ):

        emotion = detect_emotion(
            root
        )


        for filename in files:

            if not filename.lower().endswith(
                VALID_EXTENSIONS
            ):

                continue


            source = os.path.join(
                root,
                filename
            )


            # ------------------------------------------------
            # UNKNOWN EMOTION
            # ------------------------------------------------

            if emotion is None:

                ignored_images += 1

                continue


            # ------------------------------------------------
            # CORRUPT IMAGE
            # ------------------------------------------------

            if not valid_image(
                source
            ):

                print(
                    "[BAD IMAGE]",
                    source
                )

                bad_images += 1

                continue


            # ------------------------------------------------
            # DUPLICATE CHECK
            # ------------------------------------------------

            try:

                image_hash = file_hash(
                    source
                )

            except Exception:

                bad_images += 1

                continue


            if image_hash in hashes:

                duplicated_images += 1

                continue


            hashes.add(
                image_hash
            )


            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------

            extension = os.path.splitext(
                filename
            )[1].lower()


            new_filename = (

                f"{emotion}_"
                f"{counters[emotion]:07d}"
                f"{extension}"

            )


            destination = os.path.join(
                OUTPUT_DIR,
                emotion,
                new_filename
            )


            shutil.copy2(
                source,
                destination
            )


            counters[
                emotion
            ] += 1


    # ========================================================
    # RESULT
    # ========================================================

    print("")
    print("=" * 70)
    print("DATASET RESULT")
    print("=" * 70)


    total = 0


    for emotion in CLASSES:

        count = counters[
            emotion
        ]

        total += count

        print(
            f"{emotion:10s}: {count}"
        )


    print("")
    print(
        f"Total valid images : {total}"
    )

    print(
        f"Corrupt images     : {bad_images}"
    )

    print(
        f"Duplicates removed : {duplicated_images}"
    )

    print(
        f"Unknown/ignored    : {ignored_images}"
    )


    print("")
    print(
        "Prepared dataset:"
    )

    print(
        OUTPUT_DIR
    )


    if total == 0:

        print("")
        print(
            "[WARNING] No images were imported."
        )

        print(
            "The raw dataset may not contain "
            "recognizable emotion folders."
        )

    else:

        print("")
        print(
            "[SUCCESS] Dataset preparation completed."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    prepare_dataset()