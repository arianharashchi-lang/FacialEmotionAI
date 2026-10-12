import os
import shutil
import numpy as np
import cv2
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ravdess_sequences"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "cnn_actor_split"
)

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# از هر Sequence پنج فریم برای CNN برمی‌داریم
FRAME_INDICES = [
    1,
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19
]

# ============================================================
# ACTOR SPLIT
# ============================================================

TRAIN_ACTORS = {
    f"{i:02d}"
    for i in range(1, 17)
}

VAL_ACTORS = {
    f"{i:02d}"
    for i in range(17, 21)
}

TEST_ACTORS = {
    f"{i:02d}"
    for i in range(21, 25)
}


# ============================================================
# GET SPLIT
# ============================================================

def get_split(actor_id):

    if actor_id in TRAIN_ACTORS:
        return "train"

    if actor_id in VAL_ACTORS:
        return "validation"

    if actor_id in TEST_ACTORS:
        return "test"

    return None


# ============================================================
# CREATE OUTPUT
# ============================================================

def create_output():

    if os.path.exists(OUTPUT_DIR):

        print(
            "[INFO] Removing old CNN split dataset..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    for split in [
        "train",
        "validation",
        "test"
    ]:

        for emotion in EMOTIONS:

            os.makedirs(
                os.path.join(
                    OUTPUT_DIR,
                    split,
                    emotion
                ),
                exist_ok=True
            )


# ============================================================
# SAVE FRAME
# ============================================================

def save_frame(
    frame,
    split,
    emotion,
    actor_id,
    sequence_name,
    frame_number
):

    # npy frames were saved by OpenCV,
    # therefore they are already BGR.
    # cv2.imwrite preserves them correctly.

    filename = (
        f"actor_{actor_id}_"
        f"{sequence_name}_"
        f"frame_{frame_number:02d}.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        split,
        emotion,
        filename
    )

    cv2.imwrite(
        output_path,
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )


# ============================================================
# PREPARE CNN DATASET
# ============================================================

def prepare():

    print("")
    print("=" * 70)

    print(
        "CNN ACTOR-WISE DATASET PREPARATION"
    )

    print("=" * 70)


    if not os.path.isdir(
        SEQUENCE_DIR
    ):

        print(
            "[ERROR] ravdess_sequences not found:"
        )

        print(
            SEQUENCE_DIR
        )

        return


    create_output()


    counters = defaultdict(
        int
    )

    sequence_counter = defaultdict(
        int
    )

    broken_sequences = 0


    # ========================================================
    # ACTOR LOOP
    # ========================================================

    for actor_folder in sorted(
        os.listdir(SEQUENCE_DIR)
    ):

        actor_path = os.path.join(
            SEQUENCE_DIR,
            actor_folder
        )


        if not os.path.isdir(
            actor_path
        ):

            continue


        if not actor_folder.startswith(
            "actor_"
        ):

            continue


        actor_id = actor_folder.replace(
            "actor_",
            ""
        )


        split = get_split(
            actor_id
        )


        if split is None:

            continue


        print(
            f"[INFO] Actor {actor_id} -> {split}"
        )


        # ====================================================
        # EMOTION LOOP
        # ====================================================

        for emotion in EMOTIONS:

            emotion_path = os.path.join(
                actor_path,
                emotion
            )


            if not os.path.isdir(
                emotion_path
            ):

                continue


            npy_files = sorted(
                [
                    file
                    for file in os.listdir(
                        emotion_path
                    )
                    if file.lower().endswith(
                        ".npy"
                    )
                ]
            )


            for filename in npy_files:

                sequence_path = os.path.join(
                    emotion_path,
                    filename
                )


                try:

                    sequence = np.load(
                        sequence_path
                    )

                except Exception as error:

                    print(
                        "[WARNING] Could not read:",
                        sequence_path
                    )

                    print(
                        error
                    )

                    broken_sequences += 1

                    continue


                # Expected:
                # (20, 160, 160, 3)

                if (
                    sequence.ndim != 4
                    or sequence.shape[0] < 20
                    or sequence.shape[-1] != 3
                ):

                    print(
                        "[WARNING] Invalid sequence shape:",
                        filename,
                        sequence.shape
                    )

                    broken_sequences += 1

                    continue


                sequence_name = os.path.splitext(
                    filename
                )[0]


                for frame_index in FRAME_INDICES:

                    frame = sequence[
                        frame_index
                    ]


                    save_frame(
                        frame,
                        split,
                        emotion,
                        actor_id,
                        sequence_name,
                        frame_index
                    )


                    counters[
                        (
                            split,
                            emotion
                        )
                    ] += 1


                sequence_counter[
                    split
                ] += 1


    # ========================================================
    # REPORT
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "CNN DATASET COMPLETE"
    )

    print("=" * 70)


    grand_total = 0


    for split in [
        "train",
        "validation",
        "test"
    ]:

        print("")
        print(
            split.upper()
        )

        print(
            "-" * 40
        )


        split_total = 0


        for emotion in EMOTIONS:

            count = counters[
                (
                    split,
                    emotion
                )
            ]


            split_total += count


            print(
                f"{emotion:10s}: {count}"
            )


        grand_total += split_total


        print(
            f"{'TOTAL':10s}: {split_total}"
        )

        print(
            f"Sequences : "
            f"{sequence_counter[split]}"
        )


    print("")
    print(
        f"Grand total CNN images: "
        f"{grand_total}"
    )

    print(
        f"Broken sequences: "
        f"{broken_sequences}"
    )

    print("")
    print(
        "Saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    prepare()