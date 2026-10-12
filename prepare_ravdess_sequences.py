import os
import cv2
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAVDESS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ravdess_downloads"
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ravdess_sequences"
)

SEQUENCE_LENGTH = 20
IMAGE_SIZE = 160

EMOTION_MAP = {
    "01": "neutral",
    "02": None,          # calm excluded
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

os.makedirs(SEQUENCE_DIR, exist_ok=True)

# ============================================================
# FACE DETECTOR
# ============================================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV face detector.")


# ============================================================
# PARSE RAVDESS NAME
# ============================================================

def parse_filename(filename):

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    parts = name.split("-")

    if len(parts) != 7:
        return None

    emotion_code = parts[2]
    actor_code = parts[6]

    emotion = EMOTION_MAP.get(emotion_code)

    if emotion is None:
        return None

    return emotion, actor_code, name


# ============================================================
# FACE EXTRACTION
# ============================================================

def extract_face(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    margin_x = int(w * 0.15)
    margin_y = int(h * 0.20)

    height, width = frame.shape[:2]

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(
        width,
        x + w + margin_x
    )

    y2 = min(
        height,
        y + h + margin_y
    )

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return face


# ============================================================
# EXTRACT ORDERED SEQUENCE
# ============================================================

def extract_sequence(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < SEQUENCE_LENGTH:
        cap.release()
        return None

    # Keep temporal order.
    # Avoid very beginning/end of clip.
    indices = np.linspace(
        int(total_frames * 0.15),
        int(total_frames * 0.90),
        SEQUENCE_LENGTH
    ).astype(int)

    sequence = []

    last_valid_face = None

    for frame_index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        ok, frame = cap.read()

        if not ok:
            continue

        face = extract_face(frame)

        if face is None:

            # If detection misses one frame,
            # preserve sequence length with previous face.
            if last_valid_face is not None:
                face = last_valid_face.copy()
            else:
                continue

        last_valid_face = face

        sequence.append(face)

    cap.release()

    if len(sequence) != SEQUENCE_LENGTH:
        return None

    return np.stack(sequence)


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path):

    parsed = parse_filename(video_path)

    if parsed is None:
        return False

    emotion, actor, video_name = parsed

    sequence = extract_sequence(video_path)

    if sequence is None:
        print(
            "[SKIP]",
            os.path.basename(video_path)
        )
        return False

    actor_dir = os.path.join(
        SEQUENCE_DIR,
        f"actor_{actor}"
    )

    emotion_dir = os.path.join(
        actor_dir,
        emotion
    )

    os.makedirs(
        emotion_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        emotion_dir,
        video_name + ".npy"
    )

    # Shape:
    # (20, 160, 160, 3)

    np.save(
        output_path,
        sequence
    )

    return True


# ============================================================
# READ VIDEOS DIRECTLY FROM RAVDESS ZIP FILES
# ============================================================

def process_zip_files():

    import zipfile
    import tempfile

    zip_files = sorted([
        f
        for f in os.listdir(RAVDESS_DIR)
        if f.lower().endswith(".zip")
    ])

    if not zip_files:

        print(
            "[ERROR] No RAVDESS ZIP files found."
        )

        return

    print(
        f"[INFO] ZIP files found: {len(zip_files)}"
    )

    successful = 0
    skipped = 0

    emotion_counts = {
        emotion: 0
        for emotion in EMOTIONS
    }

    for zip_index, zip_name in enumerate(
        zip_files,
        start=1
    ):

        zip_path = os.path.join(
            RAVDESS_DIR,
            zip_name
        )

        print("")
        print("=" * 70)

        print(
            f"ZIP {zip_index}/{len(zip_files)}"
        )

        print(zip_name)

        print("=" * 70)

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as archive:

            video_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".mp4")
            ]

            for video_index, member in enumerate(
                video_files,
                start=1
            ):

                filename = os.path.basename(member)

                parsed = parse_filename(filename)

                if parsed is None:
                    continue

                emotion, actor, video_name = parsed

                with tempfile.TemporaryDirectory() as temp_dir:

                    archive.extract(
                        member,
                        temp_dir
                    )

                    video_path = os.path.join(
                        temp_dir,
                        member
                    )

                    success = process_video(
                        video_path
                    )

                if success:

                    successful += 1
                    emotion_counts[emotion] += 1

                else:

                    skipped += 1

                if video_index % 20 == 0:

                    print(
                        f"[PROCESS] "
                        f"{video_index}/"
                        f"{len(video_files)}"
                    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "image_size": IMAGE_SIZE,
        "channels": 3,
        "classes": EMOTIONS,
        "successful_sequences": successful,
        "skipped_sequences": skipped,
        "emotion_counts": emotion_counts,
    }

    metadata_path = os.path.join(
        SEQUENCE_DIR,
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("")
    print("=" * 70)
    print("LSTM SEQUENCE DATASET COMPLETED")
    print("=" * 70)

    for emotion in EMOTIONS:

        print(
            f"{emotion:10s}: "
            f"{emotion_counts[emotion]}"
        )

    print("")
    print(
        "Successful sequences:",
        successful
    )

    print(
        "Skipped sequences:",
        skipped
    )

    print("")
    print(
        "Sequence shape:"
    )

    print(
        f"({SEQUENCE_LENGTH}, "
        f"{IMAGE_SIZE}, "
        f"{IMAGE_SIZE}, 3)"
    )

    print("")
    print(
        "Saved to:"
    )

    print(
        SEQUENCE_DIR
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    process_zip_files()