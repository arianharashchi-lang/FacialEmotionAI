import argparse
import os
import time
from datetime import datetime

import cv2


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "anti_spoof_eval"
)

CATEGORIES = [
    "real",
    "print",
    "phone",
    "monitor",
    "replay",
]

AUTO_CAPTURE_INTERVAL = 0.60
MIN_FACE_SIZE = 80
FACE_MARGIN = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect privacy-local anti-spoofing evaluation images "
            "for one category."
        )
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=CATEGORIES,
        help="Evaluation category to collect.",
    )

    parser.add_argument(
        "--session",
        default="session01",
        help=(
            "Session name used as a subfolder, e.g. session01, "
            "daylight, lowlight."
        ),
    )

    return parser.parse_args()


def safe_crop(
    clean_frame,
    x,
    y,
    w,
    h,
    margin=FACE_MARGIN,
):
    frame_height, frame_width = (
        clean_frame.shape[:2]
    )

    x1 = max(
        0,
        x - margin,
    )

    y1 = max(
        0,
        y - margin,
    )

    x2 = min(
        frame_width,
        x + w + margin,
    )

    y2 = min(
        frame_height,
        y + h + margin,
    )

    return clean_frame[
        y1:y2,
        x1:x2
    ].copy()


def save_crop(
    output_dir,
    category,
    crop,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"{category}_{timestamp}.jpg"
    )

    path = os.path.join(
        output_dir,
        filename,
    )

    success = cv2.imwrite(
        path,
        crop,
    )

    return path if success else None


def category_instructions(category):
    instructions = {
        "real": (
            "Show a real face directly to the webcam."
        ),
        "print": (
            "Show a printed face photograph to the webcam."
        ),
        "phone": (
            "Show a static face image on a phone or tablet."
        ),
        "monitor": (
            "Show a static face image on a computer monitor."
        ),
        "replay": (
            "Play a face video on a screen and show it to the webcam."
        ),
    }

    return instructions[category]


def main():
    args = parse_args()

    output_dir = os.path.join(
        DATA_ROOT,
        args.category,
        args.session,
    )

    os.makedirs(
        output_dir,
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
            "Could not load OpenCV face detector."
        )

    camera = cv2.VideoCapture(
        0
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    print("")
    print("=" * 72)
    print("ANTI-SPOOFING ROBUSTNESS DATA COLLECTOR")
    print("=" * 72)
    print(
        f"Category : {args.category}"
    )
    print(
        f"Session  : {args.session}"
    )
    print(
        category_instructions(
            args.category
        )
    )
    print("")
    print(
        "SPACE = save one sample | A = auto capture | Q = quit"
    )
    print("")
    print(
        "Privacy: collected face images stay under ignored data/."
    )
    print(
        "Collect multiple sessions/angles/lighting conditions "
        "instead of hundreds of nearly identical frames."
    )
    print("")

    auto_capture = False
    last_capture_time = 0.0
    saved_count = 0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "[ERROR] Camera frame unavailable."
                )
                break

            frame = cv2.flip(
                frame,
                1,
            )

            clean_frame = frame.copy()

            gray = cv2.cvtColor(
                clean_frame,
                cv2.COLOR_BGR2GRAY,
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(
                    MIN_FACE_SIZE,
                    MIN_FACE_SIZE,
                ),
            )

            crops = []

            for (
                x,
                y,
                w,
                h,
            ) in faces:
                crop = safe_crop(
                    clean_frame,
                    x,
                    y,
                    w,
                    h,
                )

                if (
                    crop is not None
                    and crop.size > 0
                ):
                    crops.append(
                        crop
                    )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2,
                )

            now = time.time()

            if (
                auto_capture
                and crops
                and (
                    now
                    - last_capture_time
                    >= AUTO_CAPTURE_INTERVAL
                )
            ):
                for crop in crops:
                    path = save_crop(
                        output_dir,
                        args.category,
                        crop,
                    )

                    if path:
                        saved_count += 1

                last_capture_time = now

            status = (
                "AUTO ON"
                if auto_capture
                else "AUTO OFF"
            )

            cv2.putText(
                frame,
                (
                    f"{args.category.upper()} | "
                    f"{status} | Saved: {saved_count}"
                ),
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "SPACE=Save  A=Auto  Q=Quit",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Anti-Spoofing Robustness Collector",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == ord("a"):
                auto_capture = (
                    not auto_capture
                )

            if key == 32:
                if not crops:
                    print(
                        "[WARNING] No face detected."
                    )
                else:
                    for crop in crops:
                        path = save_crop(
                            output_dir,
                            args.category,
                            crop,
                        )

                        if path:
                            saved_count += 1
                            print(
                                "[SAVED]",
                                path,
                            )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("")
    print("=" * 72)
    print("COLLECTION FINISHED")
    print("=" * 72)
    print(
        f"Saved this run: {saved_count}"
    )
    print(
        f"Folder: {output_dir}"
    )


if __name__ == "__main__":
    main()
