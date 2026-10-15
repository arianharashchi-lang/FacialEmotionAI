from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from emotion_detector_5class import EmotionDetector5Class
from fake_detector import FakeDetector


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"


class ModelRuntime:
    """Single-user MVP runtime that reuses one loaded model stack."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError("Could not load OpenCV face detector.")

        self.fake_detector = FakeDetector()
        self.emotion_detector = EmotionDetector5Class()
        self.lock = threading.RLock()

    @staticmethod
    def _largest_face(boxes: Any) -> tuple[int, int, int, int] | None:
        if boxes is None or len(boxes) == 0:
            return None
        x, y, w, h = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(w), int(h)

    def reset(self) -> None:
        with self.lock:
            self.emotion_detector.reset()

    def analyze(self, frame: np.ndarray, force_sample: bool = False) -> dict[str, Any]:
        with self.lock:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=5,
                minSize=(70, 70),
            )

            box = self._largest_face(boxes)
            if box is None:
                return {
                    "face_detected": False,
                    "message": "No face detected.",
                    "emotion": self.emotion_detector.get_state(),
                }

            x, y, w, h = box
            pad_x = int(w * 0.12)
            pad_y = int(h * 0.12)
            frame_h, frame_w = frame.shape[:2]

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame_w, x + w + pad_x)
            y2 = min(frame_h, y + h + pad_y)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return {
                    "face_detected": False,
                    "message": "Face crop was empty.",
                    "emotion": self.emotion_detector.get_state(),
                }

            anti_spoof = self.fake_detector.predict(face)

            if anti_spoof.get("status") == "FAKE":
                return {
                    "face_detected": True,
                    "face_box": [x1, y1, x2 - x1, y2 - y1],
                    "anti_spoof": anti_spoof,
                    "message": "Frame rejected by anti-spoofing gate.",
                    "emotion": self.emotion_detector.get_state(),
                }

            if force_sample:
                self.emotion_detector.last_sample_time = 0.0

            emotion = self.emotion_detector.update(face)

            return {
                "face_detected": True,
                "face_box": [x1, y1, x2 - x1, y2 - y1],
                "anti_spoof": anti_spoof,
                "message": "ok",
                "emotion": emotion,
            }


_runtime: ModelRuntime | None = None
_runtime_init_lock = threading.Lock()


def get_runtime() -> ModelRuntime:
    global _runtime

    if _runtime is not None:
        return _runtime

    with _runtime_init_lock:
        if _runtime is None:
            _runtime = ModelRuntime()

    return _runtime


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    return frame


app = FastAPI(
    title="FacialEmotionAI Web Demo",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((TEMPLATE_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine_loaded": _runtime is not None,
    }


@app.post("/api/start")
def start_engine() -> dict[str, Any]:
    runtime = get_runtime()
    state = runtime.emotion_detector.get_state()

    return {
        "status": "ready",
        "classes": state["classes"],
        "anti_spoof_model_loaded": runtime.fake_detector.model_loaded,
    }


@app.post("/api/reset")
def reset_engine() -> dict[str, str]:
    get_runtime().reset()
    return {"status": "reset"}


@app.post("/api/frame")
async def analyze_frame(frame: UploadFile = File(...)) -> dict[str, Any]:
    image = decode_image(await frame.read())
    return get_runtime().analyze(image)


@app.post("/api/video")
async def analyze_video(video: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    runtime = get_runtime()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    capture = cv2.VideoCapture(temp_path)

    try:
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 25.0

        duration = total_frames / fps if total_frames > 0 else 0.0
        if duration > 300:
            raise HTTPException(status_code=400, detail="Maximum video length is 5 minutes.")

        target_sample_fps = 8.0
        stride = max(1, int(round(fps / target_sample_fps)))

        runtime.reset()

        timeline = []
        probability_rows = []
        confidences = []
        frame_index = 0
        sampled_with_face = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            result = runtime.analyze(frame, force_sample=True)

            if result.get("face_detected"):
                sampled_with_face += 1

            emotion = result.get("emotion") or {}
            if emotion.get("ready") and emotion.get("label"):
                probabilities = emotion.get("probabilities") or {}
                confidence = float(emotion.get("confidence") or 0.0)

                timeline.append({
                    "time": round(frame_index / fps, 2),
                    "label": emotion.get("label"),
                    "confidence": round(confidence, 4),
                    "reliability": emotion.get("reliability"),
                    "probabilities": {
                        name: round(float(value), 4)
                        for name, value in probabilities.items()
                    },
                })

                probability_rows.append({
                    name: float(value)
                    for name, value in probabilities.items()
                })
                confidences.append(confidence)

            frame_index += 1

        if not timeline:
            return {
                "status": "completed",
                "duration_seconds": round(duration, 2),
                "sampled_frames_with_face": sampled_with_face,
                "timeline": [],
                "summary": None,
                "message": "No ready temporal predictions were produced.",
            }

        classes = runtime.emotion_detector.get_state()["classes"]
        averages = {
            name: float(np.mean([row.get(name, 0.0) for row in probability_rows]))
            for name in classes
        }

        dominant = max(averages, key=averages.get)

        return {
            "status": "completed",
            "duration_seconds": round(duration, 2),
            "sampled_frames_with_face": sampled_with_face,
            "timeline": timeline,
            "summary": {
                "dominant_expression": dominant,
                "average_confidence": round(float(np.mean(confidences)), 4),
                "average_probabilities": {
                    name: round(value, 4)
                    for name, value in averages.items()
                },
            },
            "message": "Video analysis complete.",
        }

    finally:
        capture.release()
        try:
            os.remove(temp_path)
        except OSError:
            pass
