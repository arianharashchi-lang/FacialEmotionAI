import cv2
import sqlite3
import os
import time
from datetime import datetime

# مدل تشخیص Real / Fake
from fake_detector import FakeDetector


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


    last_database_log = 0


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

            minSize=(
                80,
                80
            )

        )


        face_count = len(
            faces
        )


        # ====================================================
        # NO FACE
        # ====================================================

        if face_count == 0:

            cv2.putText(

                frame,

                "NO HUMAN FACE",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 0, 255),

                2

            )


        # ====================================================
        # FACE FOUND
        # ====================================================

        else:

            for (
                x,
                y,
                w,
                h
            ) in faces:


                # ------------------------------------------------
                # Crop Face
                # ------------------------------------------------

                face_crop = frame[
                    y:y + h,
                    x:x + w
                ]


                # ------------------------------------------------
                # Run Real / Fake model
                # ------------------------------------------------

                result = fake_detector.predict(
                    face_crop
                )


                status = result[
                    "status"
                ]


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

                    real_probability = (
                        result[
                            "real_probability"
                        ]
                        * 100
                    )

                    fake_probability = (
                        result[
                            "fake_probability"
                        ]
                        * 100
                    )


                    box_color = (
                        0,
                        255,
                        0
                    )


                    label = (
                        "REAL HUMAN"
                    )


                    real_text = (
                        f"Real: "
                        f"{real_probability:.1f}%"
                    )


                    fake_text = (
                        f"Fake: "
                        f"{fake_probability:.1f}%"
                    )


                # =================================================
                # FAKE / SPOOF
                # =================================================

                else:

                    real_probability = (
                        result[
                            "real_probability"
                        ]
                        * 100
                    )

                    fake_probability = (
                        result[
                            "fake_probability"
                        ]
                        * 100
                    )


                    box_color = (
                        0,
                        0,
                        255
                    )


                    label = (
                        "FAKE / SPOOF"
                    )


                    real_text = (
                        f"Real: "
                        f"{real_probability:.1f}%"
                    )


                    fake_text = (
                        f"Fake: "
                        f"{fake_probability:.1f}%"
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


                # ------------------------------------------------
                # Status
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    label,

                    (
                        x,
                        max(
                            30,
                            y - 60
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    box_color,

                    2

                )


                # ------------------------------------------------
                # Real %
                # ------------------------------------------------

                if real_text:

                    cv2.putText(

                        frame,

                        real_text,

                        (
                            x,
                            y + h + 30
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (
                            255,
                            255,
                            255
                        ),

                        2

                    )


                # ------------------------------------------------
                # Fake %
                # ------------------------------------------------

                if fake_text:

                    cv2.putText(

                        frame,

                        fake_text,

                        (
                            x,
                            y + h + 60
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (
                            255,
                            255,
                            255
                        ),

                        2

                    )


                # =================================================
                # EMOTION PIPELINE GATE
                # =================================================

                if status == "REAL":

                    emotion_text = (
                        "Emotion Analysis: ALLOWED"
                    )

                    emotion_color = (
                        0,
                        255,
                        0
                    )

                elif status == "FAKE":

                    emotion_text = (
                        "Emotion Analysis: BLOCKED"
                    )

                    emotion_color = (
                        0,
                        0,
                        255
                    )

                else:

                    emotion_text = (
                        "Emotion Analysis: WAITING"
                    )

                    emotion_color = (
                        0,
                        255,
                        255
                    )


                cv2.putText(

                    frame,

                    emotion_text,

                    (
                        x,
                        y + h + 90
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    emotion_color,

                    2

                )


                # =================================================
                # DATABASE LOG
                # =================================================

                current_time = time.time()


                if (
                    current_time
                    - last_database_log
                    >= DATABASE_LOG_INTERVAL
                ):

                    image_path = (
                        save_face_image(
                            face_crop
                        )
                    )


                    save_detection(

                        face_detected=1,

                        human_status="HUMAN_FACE",

                        fake_status=status,

                        emotion="NOT_ANALYZED",

                        confidence=(
                            result[
                                "real_probability"
                            ]

                            if result[
                                "real_probability"
                            ]
                            is not None

                            else None
                        ),

                        image_path=image_path

                    )


                    last_database_log = (
                        current_time
                    )


        # ====================================================
        # GENERAL UI
        # ====================================================

        cv2.putText(

            frame,

            f"Detected Faces: {face_count}",

            (
                30,
                frame.shape[0] - 60
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                255,
                255,
                255
            ),

            2

        )


        cv2.putText(

            frame,

            "Q = Quit",

            (
                30,
                frame.shape[0] - 25
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (
                255,
                255,
                255
            ),

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