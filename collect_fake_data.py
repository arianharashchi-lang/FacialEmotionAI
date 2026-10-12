import cv2
import os
import time
from datetime import datetime


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

FAKE_DATASET_FOLDER = os.path.join(
    DATA_FOLDER,
    "fake_dataset",
    "fake"
)


# ============================================================
# SETTINGS
# ============================================================

AUTO_CAPTURE_INTERVAL = 0.35

MIN_FACE_SIZE = 80

FACE_MARGIN = 30


# ============================================================
# CREATE DIRECTORY
# ============================================================

os.makedirs(
    FAKE_DATASET_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD FACE DETECTOR
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
        "Could not load OpenCV face detector."
    )


# ============================================================
# COUNT EXISTING IMAGES
# ============================================================

def count_existing_images():

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    files = [

        file

        for file in os.listdir(
            FAKE_DATASET_FOLDER
        )

        if file.lower().endswith(
            valid_extensions
        )

    ]

    return len(files)


# ============================================================
# SAVE FACE
# ============================================================

def save_fake_face(
    face_image
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"fake_{timestamp}.jpg"
    )

    filepath = os.path.join(
        FAKE_DATASET_FOLDER,
        filename
    )

    success = cv2.imwrite(
        filepath,
        face_image
    )

    if success:

        return filepath

    return None


# ============================================================
# SAFE FACE CROP
# ============================================================

def crop_face_with_margin(
    frame,
    x,
    y,
    w,
    h,
    margin=FACE_MARGIN
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        frame_width,
        x + w + margin
    )

    y2 = min(
        frame_height,
        y + h + margin
    )

    face_crop = frame[
        y1:y2,
        x1:x2
    ]

    return face_crop


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 70)
    print("FAKE / SPOOF DATASET COLLECTOR")
    print("=" * 70)

    print("")
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT point the camera directly at your real face."
    )

    print(
        "Show a FACE using:"
    )

    print(
        "- Phone screen"
    )

    print(
        "- Computer monitor"
    )

    print(
        "- Printed photograph"
    )

    print(
        "- Tablet"
    )

    print(
        "- AI-generated face displayed on screen"
    )

    print("")

    print(
        "Controls:"
    )

    print(
        "SPACE = Save one fake sample"
    )

    print(
        "A = Toggle automatic capture"
    )

    print(
        "Q = Quit"
    )

    print("")


    # ========================================================
    # OPEN CAMERA
    # ========================================================

    camera = cv2.VideoCapture(
        0
    )

    if not camera.isOpened():

        print(
            "[ERROR] Could not open webcam."
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


    auto_capture = False

    last_capture_time = 0

    saved_count = count_existing_images()


    # ========================================================
    # LOOP
    # ========================================================

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "[ERROR] Camera frame unavailable."
            )

            break


        frame = cv2.flip(
            frame,
            1
        )


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_detector.detectMultiScale(

            gray,

            scaleFactor=1.1,

            minNeighbors=6,

            minSize=(
                MIN_FACE_SIZE,
                MIN_FACE_SIZE
            )

        )


        detected_face_crops = []


        # ====================================================
        # FACE LOOP
        # ====================================================

        for (
            x,
            y,
            w,
            h
        ) in faces:


            cv2.rectangle(

                frame,

                (x, y),

                (
                    x + w,
                    y + h
                ),

                (0, 0, 255),

                2

            )


            cv2.putText(

                frame,

                "FAKE DATA SAMPLE",

                (
                    x,
                    max(
                        30,
                        y - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 0, 255),

                2

            )


            face_crop = crop_face_with_margin(

                frame,

                x,
                y,
                w,
                h

            )


            if (
                face_crop is not None
                and face_crop.size > 0
            ):

                detected_face_crops.append(
                    face_crop
                )


        # ====================================================
        # AUTO CAPTURE
        # ====================================================

        if (
            auto_capture
            and detected_face_crops
        ):

            current_time = time.time()


            if (
                current_time
                - last_capture_time
                >= AUTO_CAPTURE_INTERVAL
            ):

                for face_crop in (
                    detected_face_crops
                ):

                    path = save_fake_face(
                        face_crop
                    )

                    if path:

                        saved_count += 1


                last_capture_time = (
                    current_time
                )


        # ====================================================
        # UI PANEL
        # ====================================================

        overlay = frame.copy()

        cv2.rectangle(

            overlay,

            (10, 10),

            (520, 160),

            (0, 0, 0),

            -1

        )

        frame = cv2.addWeighted(

            overlay,

            0.55,

            frame,

            0.45,

            0

        )


        # ====================================================
        # MODE
        # ====================================================

        if auto_capture:

            mode_text = (
                "AUTO CAPTURE: ON"
            )

            mode_color = (
                0,
                255,
                0
            )

        else:

            mode_text = (
                "AUTO CAPTURE: OFF"
            )

            mode_color = (
                0,
                255,
                255
            )


        cv2.putText(

            frame,

            "Fake / Spoof Dataset Collector",

            (25, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            mode_text,

            (25, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            mode_color,

            2

        )


        cv2.putText(

            frame,

            f"Saved Fake Images: {saved_count}",

            (25, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            (
                "SPACE=Save   "
                "A=Auto   "
                "Q=Quit"
            ),

            (25, 145),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1

        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(

            "Fake Dataset Collector",

            frame

        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # ----------------------------------------------------
        # Q = Exit
        # ----------------------------------------------------

        if key == ord("q"):

            break


        # ----------------------------------------------------
        # A = Auto Capture
        # ----------------------------------------------------

        elif key == ord("a"):

            auto_capture = (
                not auto_capture
            )

            print(
                "[INFO] Auto Capture:",
                auto_capture
            )


        # ----------------------------------------------------
        # SPACE = Manual Save
        # ----------------------------------------------------

        elif key == 32:

            if not detected_face_crops:

                print(
                    "[WARNING] "
                    "No face detected."
                )

            else:

                for face_crop in (
                    detected_face_crops
                ):

                    path = save_fake_face(
                        face_crop
                    )

                    if path:

                        saved_count += 1

                        print(
                            "[SAVED]",
                            path
                        )


    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


    print("")
    print("=" * 70)

    print(
        "DATA COLLECTION FINISHED"
    )

    print(
        f"Total fake images: {saved_count}"
    )

    print(
        "Saved in:"
    )

    print(
        FAKE_DATASET_FOLDER
    )

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()