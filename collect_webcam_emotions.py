import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "webcam_adaptation",
)

IMAGE_SIZE = 160
SEQUENCE_LENGTH = 20
SAMPLE_INTERVAL = 0.10

EMOTIONS = [
    "neutral",
    "sad",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect consented Sad/Neutral webcam sequences "
            "for participant-wise adapter training."
        )
    )

    parser.add_argument(
        "--person",
        required=True,
        help="Pseudonymous participant ID, e.g. p03.",
    )

    parser.add_argument(
        "--target-per-class",
        type=int,
        default=20,
        help="Recommended minimum sequences per emotion.",
    )

    return parser.parse_args()


def count_clips(
    person_dir,
    emotion,
):
    folder = os.path.join(
        person_dir,
        emotion,
    )

    if not os.path.isdir(folder):
        return 0

    return len(
        [
            filename
            for filename in os.listdir(folder)
            if filename.lower().endswith(".npy")
        ]
    )


def extract_face(
    frame,
    face_detector,
):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(90, 90),
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(
        faces,
        key=lambda box:
            box[2] * box[3],
    )

    margin_x = int(
        w * 0.12
    )

    margin_y = int(
        h * 0.15
    )

    frame_h, frame_w = (
        frame.shape[:2]
    )

    x1 = max(
        0,
        x - margin_x,
    )

    y1 = max(
        0,
        y - margin_y,
    )

    x2 = min(
        frame_w,
        x + w + margin_x,
    )

    y2 = min(
        frame_h,
        y + h + margin_y,
    )

    face = frame[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None, None

    face = cv2.resize(
        face,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # Keep BGR intentionally for compatibility with the
    # existing CNN/LSTM feature pipeline.
    return (
        face,
        (x1, y1, x2, y2),
    )


def save_sequence(
    frames,
    person_id,
    person_dir,
    emotion,
):
    sequence = np.stack(
        frames,
        axis=0,
    ).astype(
        np.uint8
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    )

    if sequence.shape != expected_shape:
        print(
            "[ERROR] Invalid sequence shape:",
            sequence.shape,
        )
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"{person_id}_"
        f"{emotion}_"
        f"{timestamp}.npy"
    )

    output_path = os.path.join(
        person_dir,
        emotion,
        filename,
    )

    np.save(
        output_path,
        sequence,
    )

    print(
        "[SAVED]",
        emotion.upper(),
        sequence.shape,
    )

    return output_path


def main():
    args = parse_args()

    person_id = args.person.strip()

    if not person_id:
        raise ValueError(
            "Participant ID cannot be empty."
        )

    if args.target_per_class < 1:
        raise ValueError(
            "--target-per-class must be at least 1."
        )

    person_dir = os.path.join(
        OUTPUT_ROOT,
        person_id,
    )

    for emotion in EMOTIONS:
        os.makedirs(
            os.path.join(
                person_dir,
                emotion,
            ),
            exist_ok=True,
        )

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(
        cascade_path
    )

    if face_detector.empty():
        raise RuntimeError(
            "Could not load Haar face detector."
        )

    camera = cv2.VideoCapture(
        0
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Webcam could not be opened."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    selected_emotion = "neutral"
    recording = False
    recorded_frames = []
    last_sample_time = 0.0

    print("")
    print("=" * 72)
    print("SAD / NEUTRAL ADAPTER DATA COLLECTION")
    print("=" * 72)
    print(
        "Participant:",
        person_id,
    )
    print(
        "Recommended target per class:",
        args.target_per_class,
    )
    print("")
    print(
        "Use a new pseudonymous person ID for each participant."
    )
    print(
        "Collect only with the participant's consent."
    )
    print(
        "Vary natural head angle and expression slightly; "
        "avoid recording many identical takes."
    )
    print("")
    print(
        "N = Neutral | S = Sad | SPACE = record | "
        "R = cancel | Q = quit"
    )

    try:
        while True:
            success, frame = (
                camera.read()
            )

            if not success:
                print(
                    "[ERROR] Cannot read camera."
                )
                break

            frame = cv2.flip(
                frame,
                1,
            )

            clean_frame = frame.copy()

            face_crop, face_box = (
                extract_face(
                    clean_frame,
                    face_detector,
                )
            )

            if face_box is not None:
                x1, y1, x2, y2 = (
                    face_box
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            if recording:
                current_time = (
                    time.monotonic()
                )

                if (
                    face_crop is not None
                    and current_time
                    - last_sample_time
                    >= SAMPLE_INTERVAL
                ):
                    recorded_frames.append(
                        face_crop.copy()
                    )

                    last_sample_time = (
                        current_time
                    )

                if (
                    len(recorded_frames)
                    >= SEQUENCE_LENGTH
                ):
                    save_sequence(
                        recorded_frames[
                            :SEQUENCE_LENGTH
                        ],
                        person_id,
                        person_dir,
                        selected_emotion,
                    )

                    recording = False
                    recorded_frames = []

            neutral_count = count_clips(
                person_dir,
                "neutral",
            )

            sad_count = count_clips(
                person_dir,
                "sad",
            )

            cv2.putText(
                frame,
                f"Person: {person_id}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Selected: {selected_emotion.upper()}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Neutral: {neutral_count}/"
                    f"{args.target_per_class}   "
                    f"Sad: {sad_count}/"
                    f"{args.target_per_class}"
                ),
                (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if recording:
                cv2.putText(
                    frame,
                    (
                        f"RECORDING "
                        f"{selected_emotion.upper()} "
                        f"{len(recorded_frames)}/"
                        f"{SEQUENCE_LENGTH}"
                    ),
                    (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Press SPACE to record",
                    (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                "N=Neutral  S=Sad",
                (25, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "SPACE=Record  R=Cancel  Q=Quit",
                (25, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Sad / Neutral Adapter Collector",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if (
                key == ord("n")
                and not recording
            ):
                selected_emotion = (
                    "neutral"
                )

            elif (
                key == ord("s")
                and not recording
            ):
                selected_emotion = (
                    "sad"
                )

            elif key == ord("r"):
                recording = False
                recorded_frames = []

                print(
                    "[INFO] Recording cancelled."
                )

            elif (
                key == 32
                and not recording
            ):
                recorded_frames = []
                last_sample_time = 0.0
                recording = True

                print(
                    "[RECORDING]",
                    selected_emotion.upper(),
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("")
    print("=" * 72)
    print("COLLECTION FINISHED")
    print("=" * 72)

    for emotion in EMOTIONS:
        print(
            f"{emotion:10s}: "
            f"{count_clips(person_dir, emotion)}"
        )


if __name__ == "__main__":
    main()
