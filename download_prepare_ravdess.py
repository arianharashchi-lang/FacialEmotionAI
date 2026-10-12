import os
import io
import zipfile
import shutil
import requests
import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DOWNLOAD_DIR = os.path.join(
    DATA_DIR,
    "ravdess_downloads"
)

TEMP_VIDEO_DIR = os.path.join(
    DATA_DIR,
    "ravdess_temp"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "emotion_rgb"
)


# ============================================================
# ZENODO
# ============================================================

ZENODO_RECORD_ID = "1188976"

ZENODO_API = (
    f"https://zenodo.org/api/records/"
    f"{ZENODO_RECORD_ID}"
)


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

# برای تست اول فقط 4 بازیگر
# بعداً 24 کن
NUMBER_OF_ACTORS = 24

# از هر ویدئو چند تصویر استخراج شود
FRAMES_PER_VIDEO = 5

IMAGE_SIZE = 224


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTION_MAP = {

    "01": "neutral",

    "02": None,       # calm را حذف می‌کنیم

    "03": "happy",

    "04": "sad",

    "05": "angry",

    "06": "fear",

    "07": "disgust",

    "08": "surprise"
}


EMOTIONS = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "sad",

    "surprise"

]


# ============================================================
# CREATE FOLDERS
# ============================================================

def create_folders():

    os.makedirs(
        DOWNLOAD_DIR,
        exist_ok=True
    )

    os.makedirs(
        TEMP_VIDEO_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for emotion in EMOTIONS:

        os.makedirs(

            os.path.join(
                OUTPUT_DIR,
                emotion
            ),

            exist_ok=True

        )


# ============================================================
# FACE DETECTOR
# ============================================================

def load_face_detector():

    cascade_path = (

        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"

    )

    detector = cv2.CascadeClassifier(
        cascade_path
    )

    if detector.empty():

        raise RuntimeError(
            "OpenCV face detector could not be loaded."
        )

    return detector


# ============================================================
# GET ZENODO FILE LIST
# ============================================================

def get_zenodo_files():

    print("")
    print("[INFO] Connecting to Zenodo...")

    response = requests.get(
        ZENODO_API,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    files = data.get(
        "files",
        []
    )

    print(
        f"[INFO] Zenodo files found: {len(files)}"
    )

    return files


# ============================================================
# FIND ACTOR ZIP
# ============================================================

def find_actor_zip(
    files,
    actor_number
):

    expected_name = (
        f"Video_Speech_Actor_"
        f"{actor_number:02d}.zip"
    )

    print(
        "[INFO] Looking for:",
        expected_name
    )

    for item in files:

        # ----------------------------------------------------
        # Get filename
        # ----------------------------------------------------

        key = item.get(
            "key",
            ""
        )

        # ----------------------------------------------------
        # Check filename
        # ----------------------------------------------------

        if key != expected_name:

            continue

        # ----------------------------------------------------
        # Get links ONLY after correct file is found
        # ----------------------------------------------------

        links = item.get(
            "links",
            {}
        )

        # ----------------------------------------------------
        # Prefer direct file content URL
        # ----------------------------------------------------

        download_url = (
            links.get("content")
            or
            links.get("self")
        )

        if not download_url:

            print(
                "[ERROR] Download URL not found for:",
                expected_name
            )

            print(
                "[DEBUG] Available links:",
                links
            )

            return None, None

        print(
            "[SUCCESS] Found:",
            expected_name
        )

        print(
            "[INFO] Download URL:",
            download_url
        )

        return (
            expected_name,
            download_url
        )

    # --------------------------------------------------------
    # File not found
    # --------------------------------------------------------

    print(
        "[ERROR] Actor ZIP was not found:",
        expected_name
    )

    return (
        None,
        None
    )

def download_file(
    url,
    destination
):

    # ========================================================
    # اگر فایل قبلاً وجود دارد، بررسی کنیم واقعاً ZIP است
    # ========================================================

    if os.path.exists(destination):

        if zipfile.is_zipfile(destination):

            print(
                "[INFO] Valid ZIP already downloaded:",
                os.path.basename(destination)
            )

            return

        else:

            print(
                "[WARNING] Existing file is corrupted/not ZIP."
            )

            print(
                "[INFO] Removing bad file..."
            )

            os.remove(destination)


    print("")
    print(
        "[DOWNLOAD]",
        os.path.basename(destination)
    )

    print(
        "[URL]",
        url
    )


    # فایل موقت
    temp_path = (
        destination + ".part"
    )


    try:

        with requests.get(
            url,
            stream=True,
            timeout=180,
            allow_redirects=True
        ) as response:

            response.raise_for_status()


            # ------------------------------------------------
            # Debug information
            # ------------------------------------------------

            content_type = response.headers.get(
                "content-type",
                ""
            )

            total_size = int(
                response.headers.get(
                    "content-length",
                    0
                )
            )


            print(
                "[INFO] Content-Type:",
                content_type
            )

            if total_size > 0:

                print(
                    "[INFO] File size:",
                    round(
                        total_size
                        / 1024
                        / 1024,
                        2
                    ),
                    "MB"
                )


            downloaded = 0


            with open(
                temp_path,
                "wb"
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue


                    file.write(
                        chunk
                    )


                    downloaded += len(
                        chunk
                    )


                    if total_size > 0:

                        percent = (
                            downloaded
                            / total_size
                            * 100
                        )

                        print(
                            f"\rDownloading: {percent:.1f}%",
                            end=""
                        )


        print("")


        # ====================================================
        # بررسی واقعی بودن ZIP
        # ====================================================

        if not zipfile.is_zipfile(
            temp_path
        ):

            print("")
            print(
                "[ERROR] Downloaded file is NOT a valid ZIP."
            )

            print(
                "[ERROR] It may be an API/HTML response."
            )


            # چند بایت اول برای تشخیص
            with open(
                temp_path,
                "rb"
            ) as file:

                preview = file.read(
                    200
                )


            print(
                "[DEBUG] Beginning of downloaded file:"
            )

            print(
                preview
            )


            os.remove(
                temp_path
            )

            raise RuntimeError(
                "Zenodo download did not return a ZIP file."
            )


        # ====================================================
        # فایل سالم است
        # ====================================================

        os.replace(
            temp_path,
            destination
        )


        print(
            "[SUCCESS] Valid ZIP downloaded:"
        )

        print(
            destination
        )


    except Exception:

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )

        raise
    
# ============================================================
# EXTRACT ZIP
# ============================================================

def extract_zip(
    zip_path,
    actor_folder
):

    print(
        "[INFO] Extracting:",
        os.path.basename(zip_path)
    )

    # اگر فولدر قبلی وجود دارد پاکش کن
    if os.path.exists(actor_folder):

        shutil.rmtree(
            actor_folder,
            ignore_errors=True
        )

    os.makedirs(
        actor_folder,
        exist_ok=True
    )

    # بررسی سالم بودن ZIP
    if not zipfile.is_zipfile(zip_path):

        raise RuntimeError(
            f"Invalid ZIP file: {zip_path}"
        )

    # استخراج فایل‌ها
    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_file:

        zip_file.extractall(
            actor_folder
        )

    print(
        "[SUCCESS] ZIP extracted."
    )

    print(
        "[INFO] Extracted to:"
    )

    print(
        actor_folder
    )
# ============================================================
# PARSE RAVDESS FILENAME
# ============================================================

def parse_ravdess_filename(
    filename
):

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]


    parts = name.split("-")


    if len(parts) != 7:

        return None


    modality = parts[0]

    vocal_channel = parts[1]

    emotion_code = parts[2]


    # فقط Video-only
    if modality != "02":

        return None


    # فقط speech
    if vocal_channel != "01":

        return None


    emotion = EMOTION_MAP.get(
        emotion_code
    )


    if emotion is None:

        return None


    return emotion


# ============================================================
# FACE CROP
# ============================================================

def detect_and_crop_face(
    frame,
    detector
):

    gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY

    )


    faces = detector.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(80, 80)

    )


    if len(faces) == 0:

        return None


    # بزرگ‌ترین صورت
    x, y, w, h = max(

        faces,

        key=lambda box:
        box[2] * box[3]

    )


    margin_x = int(
        w * 0.15
    )

    margin_y = int(
        h * 0.20
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

        return None


    face = cv2.resize(

        face,

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        interpolation=cv2.INTER_AREA

    )


    return face


# ============================================================
# EXTRACT FRAMES
# ============================================================

def extract_frames_from_video(
    video_path,
    emotion,
    detector,
    counters
):

    capture = cv2.VideoCapture(
        video_path
    )


    if not capture.isOpened():

        print(
            "[WARNING] Cannot open:",
            video_path
        )

        return


    total_frames = int(

        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

    )


    if total_frames <= 0:

        capture.release()

        return


    # قسمت‌های میانی و انتهایی کلیپ
    # معمولاً expression واضح‌تر است

    positions = np.linspace(

        0.50,

        0.92,

        FRAMES_PER_VIDEO

    )


    for position in positions:

        frame_number = int(

            total_frames
            * position

        )


        capture.set(

            cv2.CAP_PROP_POS_FRAMES,

            frame_number

        )


        success, frame = capture.read()


        if not success:

            continue


        face = detect_and_crop_face(

            frame,

            detector

        )


        if face is None:

            continue


        counters[emotion] += 1


        filename = (

            f"{emotion}_"
            f"{counters[emotion]:07d}.jpg"

        )


        output_path = os.path.join(

            OUTPUT_DIR,

            emotion,

            filename

        )


        cv2.imwrite(

            output_path,

            face,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]

        )


    capture.release()


# ============================================================
# PROCESS ACTOR
# ============================================================

def process_actor_folder(
    actor_folder,
    detector,
    counters
):

    videos = []


    for root, dirs, files in os.walk(
        actor_folder
    ):

        for filename in files:

            if filename.lower().endswith(
                ".mp4"
            ):

                videos.append(

                    os.path.join(
                        root,
                        filename
                    )

                )


    print(
        f"[INFO] Videos found: {len(videos)}"
    )


    for index, video_path in enumerate(
        videos,
        start=1
    ):

        emotion = parse_ravdess_filename(
            video_path
        )


        if emotion is None:

            continue


        extract_frames_from_video(

            video_path,

            emotion,

            detector,

            counters

        )


        if index % 20 == 0:

            print(

                f"[PROCESS] "
                f"{index}/{len(videos)}"

            )



# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 70)
    print("RAVDESS RGB EMOTION DATASET BUILDER")
    print("=" * 70)

    create_folders()

    detector = load_face_detector()

    files = get_zenodo_files()


    counters = {

        emotion: len(

            [

                name

                for name in os.listdir(

                    os.path.join(
                        OUTPUT_DIR,
                        emotion
                    )

                )

                if name.lower().endswith(
                    ".jpg"
                )

            ]

        )

        for emotion in EMOTIONS

    }

    # ========================================================
    # ACTORS
    # ========================================================

    for actor in range(
        1,
        NUMBER_OF_ACTORS + 1
    ):

        print("")
        print("=" * 70)

        print(
            f"ACTOR {actor:02d}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # Find actor ZIP on Zenodo
        # ----------------------------------------------------

        zip_name, download_url = find_actor_zip(
            files,
            actor
        )


        if zip_name is None or download_url is None:

            print(
                f"[WARNING] Actor {actor:02d} ZIP not found."
            )

            continue


        # ----------------------------------------------------
        # Local ZIP path
        # ----------------------------------------------------

        zip_path = os.path.join(
            DOWNLOAD_DIR,
            zip_name
        )


        # ----------------------------------------------------
        # Download ZIP
        # ----------------------------------------------------

        download_file(
            download_url,
            zip_path
        )


        # ----------------------------------------------------
        # Actor extraction folder
        # IMPORTANT:
        # actor_folder MUST be created BEFORE extract_zip()
        # ----------------------------------------------------

        actor_folder = os.path.join(
            TEMP_VIDEO_DIR,
            f"Actor_{actor:02d}"
        )


        print(
            "[INFO] Actor temporary folder:"
        )

        print(
            actor_folder
        )


        # ----------------------------------------------------
        # Extract ZIP
        # ----------------------------------------------------

        extract_zip(
            zip_path,
            actor_folder
        )


        # ----------------------------------------------------
        # Extract RGB face frames
        # ----------------------------------------------------

        process_actor_folder(
            actor_folder,
            detector,
            counters
        )


        # ----------------------------------------------------
        # Remove extracted temporary video files
        # Keep downloaded ZIP
        # ----------------------------------------------------

        if os.path.exists(
            actor_folder
        ):

            shutil.rmtree(
                actor_folder,
                ignore_errors=True
            )


        print(
            f"[SUCCESS] Actor {actor:02d} completed."
        )
    # ========================================================
    # REPORT
    # ========================================================

    print("")
    print("=" * 70)

    print("FINAL DATASET")

    print("=" * 70)


    total = 0


    for emotion in EMOTIONS:

        count = counters[
            emotion
        ]

        total += count


        print(

            f"{emotion:10s}: "
            f"{count}"

        )


    print("")
    print(
        f"Total RGB face images: {total}"
    )

    print("")
    print(
        "Saved in:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()