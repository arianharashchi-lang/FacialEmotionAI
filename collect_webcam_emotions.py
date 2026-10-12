import os
import cv2
import time
import argparse
import numpy as np
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "webcam_adaptation"
)

IMAGE_SIZE = 160
SEQUENCE_LENGTH = 20

# هر 0.1 ثانیه یک فریم
# 20 فریم ≈ 2 ثانیه
SAMPLE_INTERVAL = 0.10


EMOTIONS = [
    "neutral",
    "sad",
    "fear"
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--person",
    type=str,
    default="p01",
    help="Person ID, e.g. p01, p02"
)

args = parser.parse_args()

PERSON_ID = args.person


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

PERSON_DIR = os.path.join(
    OUTPUT_ROOT,
    PERSON_ID
)

for emotion in EMOTIONS:

    os.makedirs(
        os.path.join(
            PERSON_DIR,
            emotion
        ),
        exist_ok=True
    )


# ============================================================
# FACE DETECTOR
# ============================================================

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


# ============================================================
# COUNT EXISTING CLIPS
# ============================================================

def count_clips(
    emotion
):

    folder = os.path.join(
        PERSON_DIR,
        emotion
    )

    return len([
        filename
        for filename in os.listdir(folder)
        if filename.lower().endswith(".npy")
    ])


# ============================================================
# EXTRACT PRIMARY FACE
# ============================================================

def extract_face(
    frame
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(90, 90)
    )

    if len(faces) == 0:

        return None, None


    # بزرگ‌ترین صورت
    x, y, w, h = max(
        faces,
        key=lambda box:
            box[2] * box[3]
    )


    # کمی Margin
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
        x - margin_x
    )

    y1 = max(
        0,
        y - margin_y
    )

    x2 = min(
        frame_w,
        x + w + margin_x
    )

    y2 = min(
        frame_h,
        y + h + margin_y
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
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )


    # مهم:
    # عمداً BGR نگه می‌داریم چون LSTM V1 فعلی
    # با Featureهای حاصل از همین مسیر OpenCV آموزش دیده.
    return (
        face,
        (x1, y1, x2, y2)
    )


# ============================================================
# SAVE SEQUENCE
# ============================================================

def save_sequence(
    frames,
    emotion
):

    sequence = np.stack(
        frames,
        axis=0
    ).astype(
        np.uint8
    )


    expected_shape = (
        SEQUENCE_LENGTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )


    if sequence.shape != expected_shape:

        print(
            "[ERROR] Invalid sequence shape:",
            sequence.shape
        )

        return None


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )


    filename = (
        f"{PERSON_ID}_"
        f"{emotion}_"
        f"{timestamp}.npy"
    )


    output_path = os.path.join(
        PERSON_DIR,
        emotion,
        filename
    )


    np.save(
        output_path,
        sequence
    )


    print("")
    print(
        "[SAVED]",
        emotion.upper(),
        sequence.shape
    )

    print(
        output_path
    )


    return output_path


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    0
)

if not camera.isOpened():

    raise RuntimeError(
        "Webcam could not be opened."
    )


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# STATE
# ============================================================

selected_emotion = "neutral"

recording = False

recorded_frames = []

last_sample_time = 0.0


# ============================================================
# INSTRUCTIONS
# ============================================================

print("")
print("=" * 70)
print("WEBCAM EMOTION ADAPTATION DATA COLLECTION")
print("=" * 70)

print(
    "Person:",
    PERSON_ID
)

print("")
print("N = Neutral")
print("S = Sad")
print("F = Fear")
print("SPACE = Record one 20-frame sequence")
print("R = Cancel current recording")
print("Q = Quit")

print("")

for emotion in EMOTIONS:

    print(
        emotion,
        "=",
        count_clips(emotion)
    )


# ============================================================
# LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "[ERROR] Cannot read camera."
        )

        break


    # Mirror فقط برای نمایش و ذخیره
    frame = cv2.flip(
        frame,
        1
    )


    face_crop, face_box = extract_face(
        frame
    )


    # ========================================================
    # FACE BOX
    # ========================================================

    if face_box is not None:

        x1, y1, x2, y2 = (
            face_box
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


    # ========================================================
    # RECORDING
    # ========================================================

    if recording:

        current_time = time.monotonic()


        if (
            face_crop is not None
            and
            current_time
            - last_sample_time
            >= SAMPLE_INTERVAL
        ):

            recorded_frames.append(
                face_crop.copy()
            )


            last_sample_time = (
                current_time
            )


        if len(
            recorded_frames
        ) >= SEQUENCE_LENGTH:

            save_sequence(
                recorded_frames[
                    :SEQUENCE_LENGTH
                ],
                selected_emotion
            )


            recording = False

            recorded_frames = []


    # ========================================================
    # UI
    # ========================================================

    cv2.putText(
        frame,
        f"Person: {PERSON_ID}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Selected: {selected_emotion.upper()}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    neutral_count = count_clips(
        "neutral"
    )

    sad_count = count_clips(
        "sad"
    )

    fear_count = count_clips(
        "fear"
    )


    cv2.putText(
        frame,
        (
            f"Neutral: {neutral_count}   "
            f"Sad: {sad_count}   "
            f"Fear: {fear_count}"
        ),
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
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
            2
        )

    else:

        cv2.putText(
            frame,
            "Press SPACE to record",
            (25, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    cv2.putText(
        frame,
        "N=Neutral  S=Sad  F=Fear",
        (25, frame.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "SPACE=Record   R=Cancel   Q=Quit",
        (25, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Webcam Adaptation Data Collector",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key == ord("q"):

        break


    elif key == ord("n"):

        if not recording:

            selected_emotion = (
                "neutral"
            )


    elif key == ord("s"):

        if not recording:

            selected_emotion = (
                "sad"
            )


    elif key == ord("f"):

        if not recording:

            selected_emotion = (
                "fear"
            )


    elif key == ord("r"):

        recording = False

        recorded_frames = []

        print(
            "[INFO] Recording cancelled."
        )


    elif key == 32:

        if not recording:

            recorded_frames = []

            last_sample_time = 0.0

            recording = True

            print(
                "[RECORDING]",
                selected_emotion.upper()
            )


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("")
print("=" * 70)
print("COLLECTION FINISHED")
print("=" * 70)

for emotion in EMOTIONS:

    print(
        f"{emotion:10s}: "
        f"{count_clips(emotion)}"
    )