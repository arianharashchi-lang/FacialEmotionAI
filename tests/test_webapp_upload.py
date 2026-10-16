import importlib

from fastapi.testclient import TestClient


webapp = importlib.import_module("webapp.app")


def test_oversized_video_returns_413_and_removes_temp_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(webapp, "MAX_VIDEO_UPLOAD_BYTES", 10)

    original_named_temp_file = webapp.tempfile.NamedTemporaryFile

    def create_temp_file_in_test_directory(**kwargs):
        return original_named_temp_file(dir=tmp_path, **kwargs)

    monkeypatch.setattr(
        webapp.tempfile,
        "NamedTemporaryFile",
        create_temp_file_in_test_directory,
    )

    client = TestClient(webapp.app)
    response = client.post(
        "/api/video",
        files={"video": ("large.mp4", b"x" * 11, "video/mp4")},
    )

    assert response.status_code == 413
    assert "100 MB" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_unsupported_video_format_returns_400_without_temp_file(
    tmp_path, monkeypatch
):
    original_named_temp_file = webapp.tempfile.NamedTemporaryFile

    def create_temp_file_in_test_directory(**kwargs):
        return original_named_temp_file(dir=tmp_path, **kwargs)

    monkeypatch.setattr(
        webapp.tempfile,
        "NamedTemporaryFile",
        create_temp_file_in_test_directory,
    )

    client = TestClient(webapp.app)
    response = client.post(
        "/api/video",
        files={"video": ("malicious.exe", b"not a video", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported video format."
    assert list(tmp_path.iterdir()) == []




def test_frame_analysis_waits_while_video_holds_runtime_lock(monkeypatch):
    import asyncio
    import io
    import threading

    import numpy as np

    runtime = webapp.ModelRuntime.__new__(webapp.ModelRuntime)
    runtime.lock = threading.RLock()

    detector_called = threading.Event()
    frame_finished = threading.Event()

    class FakeFaceDetector:
        def detectMultiScale(self, *args, **kwargs):
            detector_called.set()
            return ()

    class FakeEmotionDetector:
        def get_state(self):
            return {}

    runtime.face_detector = FakeFaceDetector()
    runtime.emotion_detector = FakeEmotionDetector()

    monkeypatch.setattr(webapp, "get_runtime", lambda: runtime)
    monkeypatch.setattr(
        webapp,
        "decode_image",
        lambda data: np.zeros((100, 100, 3), dtype=np.uint8),
    )

    async def send_frame():
        upload = webapp.UploadFile(
            filename="frame.jpg",
            file=io.BytesIO(b"image"),
        )
        return await webapp.analyze_frame(frame=upload)

    def run_frame():
        asyncio.run(send_frame())
        frame_finished.set()

    runtime.lock.acquire()
    try:
        worker = threading.Thread(target=run_frame)
        worker.start()

        assert not detector_called.wait(timeout=0.2)
        assert not frame_finished.is_set()
    finally:
        runtime.lock.release()

    worker.join(timeout=3)
    assert not worker.is_alive()
    assert detector_called.is_set()
    assert frame_finished.is_set()
