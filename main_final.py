import cv2
import sqlite3
import os
import time
from datetime import datetime

# مدل تشخیص Real / Fake
from fake_detector import FakeDetector

# مدل نهایی احساس: CNN Feature Extractor + LSTM V1 Rebuild + Attention
from emotion_detector import EmotionDetector


# ============================================================
# CONFIG
# ============================================================

PROJECT_NAME = "Facial Emotion Recognition AI"

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

DATABASE_LOG_INTERVAL = 2

# اگر صورت برای این مدت دیده نشود، حافظه 20 فریمی پاک می‌شود.
NO_FACE_RESET_SECONDS = 1.0

# برای جلوگیری از پاک شدن Buffer با یک تشخیص اشتباه لحظه‌ای Fake
FAKE_RESET_STREAK = 3


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_project_directories():

    folders = [
        DATABASE_FOLDER,
        DATA_FOLDER,
        CAPTURED_FACES_FOLDER
    ]

    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )


# ============================================================
# DATABASE
# ============================================================

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


# ============================================================
# SAVE DETECTION
# ============================================================

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

    detection_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
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


# ============================================================
# SAVE FACE IMAGE
# ============================================================

def save_face_image(
    face_image
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"face_{timestamp}.jpg"
    )

    image_path = os.path.join(
        CAPTURED_FACES_FOLDER,
        filename
    )

    cv2.imwrite(
        image_path,
        face_image
    )

    return image_path


# ============================================================
# OPENCV FACE DETECTOR
# ============================================================

def load_face_detector():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(
        cascade_path
    )

    if face_detector.empty():

        raise RuntimeError(
            "Could not load face detector."
        )

    return face_detector


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
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_emotion_panel(
    frame,
    emotion_state,
    status
):

    panel_x = max(
        10,
        frame.shape[1] - 360
    )

    panel_y = 20
    panel_w = 340
    panel_h = 330

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (120, 120, 120),
        1
    )

    draw_text(
        frame,
        "TEMPORAL EMOTION - LSTM V1",
        panel_x + 15,
        panel_y + 30,
        (255, 255, 255),
        0.55,
        2
    )

    if status != "REAL":

        draw_text(
            frame,
            "Emotion Analysis: BLOCKED",
            panel_x + 15,
            panel_y + 70,
            (0, 0, 255),
            0.58,
            2
        )

        draw_text(
            frame,
            "Real face required",
            panel_x + 15,
            panel_y + 105,
            (190, 190, 190),
            0.52,
            1
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

    if not ready:

        draw_text(
            frame,
            f"Collecting frames: {collected}/{required}",
            panel_x + 15,
            panel_y + 72,
            (0, 255, 255),
            0.60,
            2
        )

        progress_width = int(
            300
            * min(
                1.0,
                collected / max(required, 1)
            )
        )

        cv2.rectangle(
            frame,
            (panel_x + 15, panel_y + 95),
            (panel_x + 315, panel_y + 115),
            (90, 90, 90),
            1
        )

        cv2.rectangle(
            frame,
            (panel_x + 15, panel_y + 95),
            (panel_x + 15 + progress_width, panel_y + 115),
            (0, 255, 255),
            -1
        )

        draw_text(
            frame,
            "Keep your face visible...",
            panel_x + 15,
            panel_y + 150,
            (210, 210, 210),
            0.52,
            1
        )

        return

    label = emotion_state.get(
        "label"
    ) or "UNKNOWN"

    confidence = float(
        emotion_state.get(
            "confidence",
            0.0
        )
    )

    draw_text(
        frame,
        f"Dominant: {label.upper()}  {confidence * 100:.1f}%",
        panel_x + 15,
        panel_y + 68,
        (0, 255, 0),
        0.62,
        2
    )

    probabilities = emotion_state.get(
        "probabilities",
        {}
    )

    display_order = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]

    line_y = panel_y + 108

    for emotion in display_order:

        probability = float(
            probabilities.get(
                emotion,
                0.0
            )
        )

        draw_text(
            frame,
            f"{emotion.capitalize():9s} {probability * 100:5.1f}%",
            panel_x + 20,
            line_y,
            (235, 235, 235),
            0.54,
            1
        )

        line_y += 30


# ============================================================
# MAIN
# ============================================================

def start_camera():

    print("")
    print("=" * 60)
    print(PROJECT_NAME)
    print("=" * 60)

    # --------------------------------------------------------
    # OpenCV Face Detector
    # --------------------------------------------------------

    face_detector = load_face_detector()

    # --------------------------------------------------------
    # Load Real / Fake AI Model
    # --------------------------------------------------------

    print("")
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

    # --------------------------------------------------------
    # Load Emotion AI
    # --------------------------------------------------------

    print("")
    print(
        "[INFO] Loading CNN + LSTM V1 emotion pipeline..."
    )

    emotion_detector = EmotionDetector()

    print(
        "[SUCCESS] Emotion pipeline loaded."
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

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
    last_face_seen_time = time.monotonic()
    fake_streak = 0

    # آخرین وضعیت Emotion برای UI
    emotion_state = emotion_detector.get_state()
    primary_status = "NO_FACE"

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "[ERROR] Cannot read camera frame."
            )

            break

        # Mirror
        frame = cv2.flip(
            frame,
            1
        )

        # Convert to gray
        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80)
        )

        face_count = len(
            faces
        )

        current_monotonic = time.monotonic()

        # ====================================================
        # NO FACE
        # ====================================================

        if face_count == 0:

            primary_status = "NO_FACE"
            fake_streak = 0

            draw_text(
                frame,
                "NO HUMAN FACE",
                30,
                50,
                (0, 0, 255),
                1.0,
                2
            )

            # یک قطع کوتاه Face Detection باعث نابودی سریع Buffer نشود.
            if (
                current_monotonic
                - last_face_seen_time
                >= NO_FACE_RESET_SECONDS
            ):

                if len(
                    emotion_detector.feature_buffer
                ) > 0:

                    emotion_detector.reset()

                emotion_state = emotion_detector.get_state()

        # ====================================================
        # FACE FOUND
        # ====================================================

        else:

            last_face_seen_time = current_monotonic

            # بزرگ‌ترین صورت به عنوان Primary انتخاب می‌شود.
            primary_index = max(
                range(face_count),
                key=lambda index:
                    faces[index][2]
                    * faces[index][3]
            )

            for face_index, (
                x,
                y,
                w,
                h
            ) in enumerate(faces):

                is_primary = (
                    face_index
                    == primary_index
                )

                # ------------------------------------------------
                # Crop Face
                # ------------------------------------------------

                face_crop = frame[
                    y:y + h,
                    x:x + w
                ]

                if face_crop.size == 0:
                    continue

                # ------------------------------------------------
                # Run Real / Fake model
                # ------------------------------------------------

                result = fake_detector.predict(
                    face_crop
                )

                status = result.get(
                    "status",
                    "MODEL_NOT_TRAINED"
                )

                real_probability_raw = result.get(
                    "real_probability"
                )

                fake_probability_raw = result.get(
                    "fake_probability"
                )

                real_probability = (
                    float(real_probability_raw)
                    * 100
                    if real_probability_raw is not None
                    else None
                )

                fake_probability = (
                    float(fake_probability_raw)
                    * 100
                    if fake_probability_raw is not None
                    else None
                )

                # =================================================
                # MODEL NOT TRAINED / NOT FOUND
                # =================================================

                if status == "MODEL_NOT_TRAINED":

                    box_color = (
                        0,
                        255,
                        255
                    )

                    label = (
                        "MODEL NOT READY"
                    )

                    real_text = ""
                    fake_text = ""

                # =================================================
                # REAL FACE
                # =================================================

                elif status == "REAL":

                    box_color = (
                        0,
                        255,
                        0
                    )

                    label = (
                        "REAL HUMAN"
                    )

                    real_text = (
                        f"Real: {real_probability:.1f}%"
                        if real_probability is not None
                        else "Real: N/A"
                    )

                    fake_text = (
                        f"Fake: {fake_probability:.1f}%"
                        if fake_probability is not None
                        else "Fake: N/A"
                    )

                # =================================================
                # FAKE / SPOOF
                # =================================================

                else:

                    box_color = (
                        0,
                        0,
                        255
                    )

                    label = (
                        "FAKE / SPOOF"
                    )

                    real_text = (
                        f"Real: {real_probability:.1f}%"
                        if real_probability is not None
                        else "Real: N/A"
                    )

                    fake_text = (
                        f"Fake: {fake_probability:.1f}%"
                        if fake_probability is not None
                        else "Fake: N/A"
                    )

                # ------------------------------------------------
                # Bounding Box
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (
                        x + w,
                        y + h
                    ),
                    box_color,
                    3
                )

                # Primary marker
                if is_primary:

                    cv2.rectangle(
                        frame,
                        (x - 3, y - 3),
                        (x + w + 3, y + h + 3),
                        (255, 255, 255),
                        1
                    )

                # ------------------------------------------------
                # Status
                # ------------------------------------------------

                draw_text(
                    frame,
                    label,
                    x,
                    max(
                        30,
                        y - 60
                    ),
                    box_color,
                    0.8,
                    2
                )

                if is_primary:

                    draw_text(
                        frame,
                        "PRIMARY",
                        x,
                        max(
                            50,
                            y - 35
                        ),
                        (255, 255, 255),
                        0.48,
                        1
                    )

                # ------------------------------------------------
                # Real %
                # ------------------------------------------------

                if real_text:

                    draw_text(
                        frame,
                        real_text,
                        x,
                        y + h + 30,
                        (255, 255, 255),
                        0.65,
                        2
                    )

                # ------------------------------------------------
                # Fake %
                # ------------------------------------------------

                if fake_text:

                    draw_text(
                        frame,
                        fake_text,
                        x,
                        y + h + 60,
                        (255, 255, 255),
                        0.65,
                        2
                    )

                # =================================================
                # EMOTION PIPELINE - ONLY PRIMARY FACE
                # =================================================

                if is_primary:

                    primary_status = status

                    if status == "REAL":

                        fake_streak = 0

                        try:

                            emotion_state = emotion_detector.update(
                                face_crop
                            )

                        except Exception as error:

                            print(
                                "[ERROR] Emotion inference failed:",
                                error
                            )

                            emotion_detector.reset()
                            emotion_state = emotion_detector.get_state()

                        if emotion_state.get(
                            "ready",
                            False
                        ):

                            emotion_label = emotion_state.get(
                                "label",
                                "UNKNOWN"
                            )

                            emotion_confidence = float(
                                emotion_state.get(
                                    "confidence",
                                    0.0
                                )
                            )

                            emotion_text = (
                                f"Emotion: "
                                f"{emotion_label.upper()} "
                                f"{emotion_confidence * 100:.1f}%"
                            )

                        else:

                            emotion_text = (
                                "Emotion frames: "
                                f"{emotion_state.get('collected', 0)}"
                                "/"
                                f"{emotion_state.get('required', 20)}"
                            )

                        emotion_color = (
                            0,
                            255,
                            0
                        )

                    elif status == "FAKE":

                        fake_streak += 1

                        if fake_streak >= FAKE_RESET_STREAK:

                            emotion_detector.reset()
                            emotion_state = emotion_detector.get_state()

                        emotion_text = (
                            "Emotion Analysis: BLOCKED"
                        )

                        emotion_color = (
                            0,
                            0,
                            255
                        )

                    else:

                        fake_streak = 0

                        emotion_text = (
                            "Emotion Analysis: WAITING"
                        )

                        emotion_color = (
                            0,
                            255,
                            255
                        )

                    draw_text(
                        frame,
                        emotion_text,
                        x,
                        y + h + 90,
                        emotion_color,
                        0.55,
                        2
                    )

                    # =================================================
                    # DATABASE LOG - PRIMARY FACE ONLY
                    # =================================================

                    current_time = time.time()

                    if (
                        current_time
                        - last_database_log
                        >= DATABASE_LOG_INTERVAL
                    ):

                        image_path = save_face_image(
                            face_crop
                        )

                        if (
                            status == "REAL"
                            and emotion_state.get(
                                "ready",
                                False
                            )
                        ):

                            database_emotion = emotion_state.get(
                                "label",
                                "UNKNOWN"
                            )

                            database_confidence = float(
                                emotion_state.get(
                                    "confidence",
                                    0.0
                                )
                            )

                        elif status == "REAL":

                            database_emotion = "COLLECTING"

                            database_confidence = (
                                float(real_probability_raw)
                                if real_probability_raw is not None
                                else None
                            )

                        elif status == "FAKE":

                            database_emotion = "BLOCKED"

                            database_confidence = (
                                float(fake_probability_raw)
                                if fake_probability_raw is not None
                                else None
                            )

                        else:

                            database_emotion = "NOT_ANALYZED"
                            database_confidence = None

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

                else:

                    draw_text(
                        frame,
                        "Emotion: PRIMARY FACE ONLY",
                        x,
                        y + h + 90,
                        (180, 180, 180),
                        0.48,
                        1
                    )

        # ====================================================
        # EMOTION SIDE PANEL
        # ====================================================

        draw_emotion_panel(
            frame,
            emotion_state,
            primary_status
        )

        # ====================================================
        # GENERAL UI
        # ====================================================

        draw_text(
            frame,
            f"Detected Faces: {face_count}",
            30,
            frame.shape[0] - 60,
            (255, 255, 255),
            0.65,
            2
        )

        draw_text(
            frame,
            "Q = Quit",
            30,
            frame.shape[0] - 25,
            (255, 255, 255),
            0.6,
            1
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            PROJECT_NAME,
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()
    cv2.destroyAllWindows()


# ============================================================
# START PROGRAM
# ============================================================

def main():

    create_project_directories()
    initialize_database()
    start_camera()


if __name__ == "__main__":

    main()
