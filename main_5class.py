import os
import time
import sqlite3
from datetime import datetime

import cv2
import numpy as np

from fake_detector import FakeDetector
from emotion_detector_5class import EmotionDetector5Class


# ============================================================
# CONFIG
# ============================================================

PROJECT_NAME = "Facial Emotion Recognition AI - Final 5 Class"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE_FOLDER = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_FOLDER,
    "emotion_ai.db"
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

CAPTURED_FACES_FOLDER = os.path.join(
    DATA_FOLDER,
    "captured_faces"
)

DATABASE_LOG_INTERVAL = 2.0
NO_FACE_RESET_SECONDS = 1.0
FAKE_RESET_STREAK = 3
PANEL_WIDTH = 420


# ============================================================
# DIRECTORIES / DATABASE
# ============================================================

def create_project_directories():

    for folder in [
        DATABASE_FOLDER,
        DATA_FOLDER,
        CAPTURED_FACES_FOLDER
    ]:

        os.makedirs(
            folder,
            exist_ok=True
        )


def initialize_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detections
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_time TEXT NOT NULL,
            face_detected INTEGER NOT NULL,
            human_status TEXT,
            fake_status TEXT,
            emotion TEXT,
            confidence REAL,
            image_path TEXT
        )
        """
    )

    connection.commit()
    connection.close()

    print(
        "[INFO] Database ready."
    )


def save_detection(
    face_detected,
    human_status,
    fake_status=None,
    emotion=None,
    confidence=None,
    image_path=None
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    detection_time = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    cursor.execute(
        """
        INSERT INTO detections
        (
            detection_time,
            face_detected,
            human_status,
            fake_status,
            emotion,
            confidence,
            image_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            detection_time,
            face_detected,
            human_status,
            fake_status,
            emotion,
            confidence,
            image_path
        )
    )

    connection.commit()
    connection.close()


def save_face_image(
    face_image
):

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    path = os.path.join(
        CAPTURED_FACES_FOLDER,
        f"face_{timestamp}.jpg"
    )

    cv2.imwrite(
        path,
        face_image
    )

    return path


# ============================================================
# FACE DETECTOR
# ============================================================

def load_face_detector():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = (
        cv2.CascadeClassifier(
            cascade_path
        )
    )

    if detector.empty():

        raise RuntimeError(
            "Could not load face detector."
        )

    return detector


# ============================================================
# UI HELPERS
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.60,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        (
            int(x),
            int(y)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def make_output_canvas(
    frame
):

    h, w = frame.shape[:2]

    canvas = np.zeros(
        (
            h,
            w + PANEL_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :,
        :w
    ] = frame

    canvas[
        :,
        w:
    ] = (
        24,
        24,
        24
    )

    return (
        canvas,
        w
    )


def draw_emotion_panel(
    canvas,
    panel_x,
    emotion_state,
    status
):

    x = (
        panel_x
        + 18
    )

    y = 34

    draw_text(
        canvas,
        "FINAL TEMPORAL EMOTION - 5 CLASS",
        x,
        y,
        (
            255,
            255,
            255
        ),
        0.52,
        2
    )

    y += 34

    draw_text(
        canvas,
        "Disgust->Sad | Fear->Surprise",
        x,
        y,
        (
            180,
            180,
            180
        ),
        0.43,
        1
    )

    y += 38

    if status != "REAL":

        draw_text(
            canvas,
            "Emotion Analysis: BLOCKED",
            x,
            y,
            (
                0,
                0,
                255
            ),
            0.57,
            2
        )

        return

    collected = emotion_state.get(
        "collected",
        0
    )

    required = emotion_state.get(
        "required",
        20
    )

    ready = emotion_state.get(
        "ready",
        False
    )

    draw_text(
        canvas,
        f"Temporal Buffer: {collected}/{required}",
        x,
        y,
        (
            210,
            210,
            210
        ),
        0.50,
        1
    )

    y += 32

    if not ready:

        draw_text(
            canvas,
            "Collecting temporal frames...",
            x,
            y,
            (
                0,
                255,
                255
            ),
            0.52,
            2
        )

        return

    label = (
        emotion_state.get(
            "label"
        )
        or
        "UNKNOWN"
    ).upper()

    confidence = float(
        emotion_state.get(
            "confidence",
            0.0
        )
    )

    margin = float(
        emotion_state.get(
            "margin",
            0.0
        )
    )

    reliability = emotion_state.get(
        "reliability",
        "LOW"
    )

    if reliability == "HIGH":

        reliability_color = (
            0,
            255,
            0
        )

    elif reliability == "MEDIUM":

        reliability_color = (
            0,
            220,
            255
        )

    else:

        reliability_color = (
            0,
            120,
            255
        )

    draw_text(
        canvas,
        f"Dominant: {label}",
        x,
        y,
        (
            0,
            255,
            0
        ),
        0.63,
        2
    )

    y += 34

    draw_text(
        canvas,
        f"Confidence: {confidence * 100:.1f}%",
        x,
        y,
        (
            235,
            235,
            235
        ),
        0.51,
        1
    )

    y += 27

    draw_text(
        canvas,
        f"Reliability: {reliability}",
        x,
        y,
        reliability_color,
        0.50,
        2
    )

    y += 26

    draw_text(
        canvas,
        f"Top-2 margin: {margin * 100:.1f}%",
        x,
        y,
        (
            180,
            180,
            180
        ),
        0.43,
        1
    )

    y += 36

    probabilities = emotion_state.get(
        "probabilities",
        {}
    )

    for emotion in [
        "angry",
        "happy",
        "surprise",
        "sad",
        "neutral"
    ]:

        probability = float(
            probabilities.get(
                emotion,
                0.0
            )
        )

        color = (
            255,
            255,
            255
        )

        if emotion.upper() == label:

            color = (
                0,
                255,
                0
            )

        draw_text(
            canvas,
            (
                f"{emotion.capitalize():9s} "
                f"{probability * 100:5.1f}%"
            ),
            x,
            y,
            color,
            0.50,
            1
        )

        y += 30

    y += 8

    adapter_active = bool(
        emotion_state.get(
            "adapter_active",
            False
        )
    )

    adapter_confidence = float(
        emotion_state.get(
            "adapter_confidence",
            0.0
        )
    )

    adapter_text = (
        "Sad/Neutral adapter: ACTIVE"
        if adapter_active
        else
        "Sad/Neutral adapter: BYPASS"
    )

    adapter_color = (
        (
            0,
            255,
            0
        )
        if adapter_active
        else
        (
            150,
            150,
            150
        )
    )

    draw_text(
        canvas,
        adapter_text,
        x,
        y,
        adapter_color,
        0.45,
        1
    )

    y += 25

    draw_text(
        canvas,
        (
            f"Adapter confidence: "
            f"{adapter_confidence * 100:.1f}%"
        ),
        x,
        y,
        (
            180,
            180,
            180
        ),
        0.42,
        1
    )

    y += 24

    adapter_threshold = float(
        emotion_state.get(
            "adapter_threshold",
            0.70
        )
    )

    draw_text(
        canvas,
        (
            f"Adapter gate: "
            f"{adapter_threshold * 100:.1f}%"
        ),
        x,
        y,
        (
            160,
            160,
            160
        ),
        0.40,
        1
    )

    y += 26

    adapter_probs = emotion_state.get(
        "adapter_probabilities",
        {}
    )

    if adapter_probs:

        draw_text(
            canvas,
            (
                f"Neutral "
                f"{adapter_probs.get('neutral', 0.0) * 100:.1f}%"
                f" | Sad "
                f"{adapter_probs.get('sad', 0.0) * 100:.1f}%"
            ),
            x,
            y,
            (
                200,
                200,
                200
            ),
            0.42,
            1
        )

    y += 34

    draw_text(
        canvas,
        "R = reset buffer   Q = quit",
        x,
        y,
        (
            170,
            170,
            170
        ),
        0.42,
        1
    )


# ============================================================
# CAMERA
# ============================================================

def start_camera():

    print("")
    print("=" * 64)
    print(PROJECT_NAME)
    print("=" * 64)

    face_detector = (
        load_face_detector()
    )

    print(
        "[INFO] Loading Real/Fake model..."
    )

    fake_detector = FakeDetector(
        real_threshold=0.65
    )

    if fake_detector.model_loaded:

        print(
            "[SUCCESS] Real/Fake model loaded."
        )

    else:

        print(
            "[WARNING] Real/Fake model NOT loaded."
        )

    print(
        "[INFO] Loading final 5-class emotion pipeline..."
    )

    emotion_detector = (
        EmotionDetector5Class()
    )

    print(
        "[SUCCESS] Final emotion pipeline loaded."
    )

    camera = cv2.VideoCapture(
        0
    )

    if not camera.isOpened():

        print(
            "[ERROR] Webcam could not be opened."
        )

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    last_database_log = 0.0

    last_face_seen_time = (
        time.monotonic()
    )

    fake_streak = 0

    emotion_state = (
        emotion_detector.get_state()
    )

    primary_status = (
        "NO_FACE"
    )

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        success, frame = (
            camera.read()
        )

        if not success:

            print(
                "[ERROR] Cannot read camera frame."
            )

            break

        frame = cv2.flip(
            frame,
            1
        )

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(
                80,
                80
            )
        )

        face_count = len(
            faces
        )

        now_mono = (
            time.monotonic()
        )

        if face_count == 0:

            primary_status = (
                "NO_FACE"
            )

            fake_streak = 0

            draw_text(
                frame,
                "NO HUMAN FACE",
                30,
                50,
                (
                    0,
                    0,
                    255
                ),
                1.0,
                2
            )

            if (
                now_mono
                - last_face_seen_time
                >=
                NO_FACE_RESET_SECONDS
            ):

                if len(
                    emotion_detector.feature_buffer
                ) > 0:

                    emotion_detector.reset()

                emotion_state = (
                    emotion_detector.get_state()
                )

        else:

            last_face_seen_time = (
                now_mono
            )

            primary_index = max(
                range(
                    face_count
                ),
                key=lambda index:
                    faces[
                        index
                    ][2]
                    *
                    faces[
                        index
                    ][3]
            )

            for face_index, (
                x,
                y,
                w,
                h
            ) in enumerate(
                faces
            ):

                is_primary = (
                    face_index
                    ==
                    primary_index
                )

                face_crop = frame[
                    y:y + h,
                    x:x + w
                ]

                if face_crop.size == 0:
                    continue

                result = fake_detector.predict(
                    face_crop
                )

                status = result.get(
                    "status",
                    "MODEL_NOT_TRAINED"
                )

                real_raw = result.get(
                    "real_probability"
                )

                fake_raw = result.get(
                    "fake_probability"
                )

                real_pct = (
                    float(
                        real_raw
                    )
                    * 100
                    if real_raw is not None
                    else None
                )

                fake_pct = (
                    float(
                        fake_raw
                    )
                    * 100
                    if fake_raw is not None
                    else None
                )

                if status == "REAL":

                    box_color = (
                        0,
                        255,
                        0
                    )

                    label = (
                        "REAL HUMAN"
                    )

                elif status == "FAKE":

                    box_color = (
                        0,
                        0,
                        255
                    )

                    label = (
                        "FAKE / SPOOF"
                    )

                else:

                    box_color = (
                        0,
                        255,
                        255
                    )

                    label = (
                        "MODEL NOT READY"
                    )

                cv2.rectangle(
                    frame,
                    (
                        x,
                        y
                    ),
                    (
                        x + w,
                        y + h
                    ),
                    box_color,
                    3
                )

                draw_text(
                    frame,
                    label,
                    x,
                    max(
                        30,
                        y - 42
                    ),
                    box_color,
                    0.78,
                    2
                )

                if is_primary:

                    draw_text(
                        frame,
                        "PRIMARY",
                        x,
                        max(
                            50,
                            y - 18
                        ),
                        (
                            255,
                            255,
                            255
                        ),
                        0.45,
                        1
                    )

                info_y = min(
                    frame.shape[0]
                    - 95,
                    y
                    + h
                    + 28
                )

                if real_pct is not None:

                    draw_text(
                        frame,
                        f"Real: {real_pct:.1f}%",
                        x,
                        info_y,
                        (
                            255,
                            255,
                            255
                        ),
                        0.58,
                        2
                    )

                if fake_pct is not None:

                    draw_text(
                        frame,
                        f"Fake: {fake_pct:.1f}%",
                        x,
                        info_y + 26,
                        (
                            255,
                            255,
                            255
                        ),
                        0.58,
                        2
                    )

                if not is_primary:
                    continue

                primary_status = (
                    status
                )

                if status == "REAL":

                    fake_streak = 0

                    try:

                        emotion_state = (
                            emotion_detector.update(
                                face_crop
                            )
                        )

                    except Exception as error:

                        print(
                            "[ERROR] Emotion inference failed:",
                            error
                        )

                        emotion_detector.reset()

                        emotion_state = (
                            emotion_detector.get_state()
                        )

                    if emotion_state.get(
                        "ready",
                        False
                    ):

                        emotion_label = (
                            emotion_state.get(
                                "label",
                                "UNKNOWN"
                            )
                        )

                        emotion_confidence = float(
                            emotion_state.get(
                                "confidence",
                                0.0
                            )
                        )

                        reliability = (
                            emotion_state.get(
                                "reliability",
                                "LOW"
                            )
                        )

                        draw_text(
                            frame,
                            (
                                f"Emotion: "
                                f"{emotion_label.upper()} "
                                f"{emotion_confidence * 100:.1f}% "
                                f"({reliability})"
                            ),
                            x,
                            info_y + 54,
                            (
                                0,
                                255,
                                0
                            ),
                            0.49,
                            2
                        )

                    else:

                        draw_text(
                            frame,
                            (
                                f"Emotion frames: "
                                f"{emotion_state.get('collected', 0)}/"
                                f"{emotion_state.get('required', 20)}"
                            ),
                            x,
                            info_y + 54,
                            (
                                0,
                                255,
                                255
                            ),
                            0.48,
                            2
                        )

                elif status == "FAKE":

                    fake_streak += 1

                    if (
                        fake_streak
                        >=
                        FAKE_RESET_STREAK
                    ):

                        emotion_detector.reset()

                        emotion_state = (
                            emotion_detector.get_state()
                        )

                else:

                    fake_streak = 0

                current_time = (
                    time.time()
                )

                if (
                    current_time
                    - last_database_log
                    >=
                    DATABASE_LOG_INTERVAL
                ):

                    image_path = (
                        save_face_image(
                            face_crop
                        )
                    )

                    if (
                        status == "REAL"
                        and
                        emotion_state.get(
                            "ready",
                            False
                        )
                    ):

                        database_emotion = (
                            emotion_state.get(
                                "label",
                                "UNKNOWN"
                            )
                        )

                        database_confidence = float(
                            emotion_state.get(
                                "confidence",
                                0.0
                            )
                        )

                    elif status == "REAL":

                        database_emotion = (
                            "COLLECTING"
                        )

                        database_confidence = (
                            float(
                                real_raw
                            )
                            if real_raw is not None
                            else None
                        )

                    elif status == "FAKE":

                        database_emotion = (
                            "BLOCKED"
                        )

                        database_confidence = (
                            float(
                                fake_raw
                            )
                            if fake_raw is not None
                            else None
                        )

                    else:

                        database_emotion = (
                            "NOT_ANALYZED"
                        )

                        database_confidence = (
                            None
                        )

                    save_detection(
                        face_detected=1,
                        human_status="HUMAN_FACE",
                        fake_status=status,
                        emotion=database_emotion,
                        confidence=database_confidence,
                        image_path=image_path
                    )

                    last_database_log = (
                        current_time
                    )

        draw_text(
            frame,
            f"Detected Faces: {face_count}",
            20,
            frame.shape[0] - 42,
            (
                255,
                255,
                255
            ),
            0.56,
            2
        )

        draw_text(
            frame,
            "R = Reset   Q = Quit",
            20,
            frame.shape[0] - 16,
            (
                255,
                255,
                255
            ),
            0.50,
            1
        )

        canvas, panel_x = (
            make_output_canvas(
                frame
            )
        )

        draw_emotion_panel(
            canvas,
            panel_x,
            emotion_state,
            primary_status
        )

        cv2.imshow(
            PROJECT_NAME,
            canvas
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

        if key == ord("r"):

            emotion_detector.reset()

            emotion_state = (
                emotion_detector.get_state()
            )

            print(
                "[INFO] Temporal buffer reset."
            )

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

def main():

    create_project_directories()

    initialize_database()

    start_camera()


if __name__ == "__main__":

    main()
