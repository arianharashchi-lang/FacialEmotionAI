from __future__ import annotations

import copy
import csv
import io
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webapp.app import ModelRuntime, _analyze_video_path, get_runtime  # noqa: E402


st.set_page_config(
    page_title="FacialEmotionAI",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(circle at 12% 0%, #10253a 0, #07111f 34%, #050b14 100%); }
      h1, h2, h3 { letter-spacing: -0.035em; }
      .hero { padding: 0.7rem 0 1.2rem 0; }
      .muted { color: #9fb2c8; }
      div[data-testid="stMetric"] { background: rgba(12,25,43,.72); border: 1px solid rgba(137,186,255,.18); padding: 1rem; border-radius: 14px; }
      .stButton button, .stDownloadButton button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

FINAL_CLASSES = ["angry", "happy", "surprise", "sad", "neutral"]
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_BYTES = 100 * 1024 * 1024


@st.cache_resource(show_spinner=False)
def _base_runtime() -> ModelRuntime:
    """Load model weights once per app process."""
    return get_runtime()


def _session_runtime() -> ModelRuntime:
    """Share loaded weights while keeping temporal state separate per browser session."""
    if "facialemotionai_runtime" not in st.session_state:
        base = _base_runtime()
        runtime = copy.copy(base)
        detector = copy.copy(base.emotion_detector)
        detector.feature_buffer = deque(
            maxlen=getattr(base.emotion_detector.feature_buffer, "maxlen", 20) or 20
        )
        detector.reset()
        runtime.emotion_detector = detector
        # Keep the base lock so model inference stays serialized across sessions.
        runtime.lock = base.lock
        st.session_state["facialemotionai_runtime"] = runtime
    return st.session_state["facialemotionai_runtime"]


class EmotionVideoProcessor(VideoProcessorBase):
    """Process a browser camera stream and annotate sampled frames."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self.frame_index = 0
        self._result_lock = threading.Lock()
        self._latest_result: dict[str, Any] | None = None
        self._latest_error: str | None = None

    def get_latest(self) -> tuple[dict[str, Any] | None, str | None]:
        with self._result_lock:
            result = copy.deepcopy(self._latest_result)
            return result, self._latest_error

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_index += 1

        # Sampling every fourth browser frame limits CPU use while preserving
        # enough temporal samples for the model's 20-frame sequence.
        if self.frame_index % 4 == 0:
            try:
                result = self.runtime.analyze(image)
                with self._result_lock:
                    self._latest_result = result
                    self._latest_error = None
            except Exception as exc:  # keep the camera preview alive on inference errors
                with self._result_lock:
                    self._latest_error = str(exc)

        result, _ = self.get_latest()
        if result and result.get("face_detected"):
            box = result.get("face_box")
            if box and len(box) == 4:
                x, y, width, height = (int(value) for value in box)
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + width, y + height),
                    (80, 215, 255),
                    2,
                )

            emotion = result.get("emotion") or {}
            if emotion.get("ready") and emotion.get("label"):
                label = str(emotion["label"]).upper()
            else:
                collected = int(emotion.get("collected") or 0)
                required = int(emotion.get("required") or 20)
                label = f"COLLECTING {collected}/{required}"

            cv2.putText(
                image,
                label,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 215, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


def _render_live_result(result: dict[str, Any] | None, error: str | None) -> None:
    if error:
        st.error(f"Inference error: {error}")
        return
    if not result:
        st.info("Waiting for the first sampled frame…")
        return
    if not result.get("face_detected"):
        st.warning("No face detected. Center your face in the camera.")
        return

    anti_spoof = result.get("anti_spoof") or {}
    emotion = result.get("emotion") or {}
    status = anti_spoof.get("status", "NOT AVAILABLE")
    st.caption(f"Real/Fake screening: **{status}**")

    if status == "FAKE":
        st.warning("This frame was rejected by the project's anti-spoofing gate.")
        return
    if not emotion.get("ready"):
        collected = int(emotion.get("collected") or 0)
        required = int(emotion.get("required") or 20)
        st.info(f"Building temporal context: {collected}/{required} samples")
        return

    col1, col2 = st.columns(2)
    col1.metric("Expression label", str(emotion.get("label", "—")).title())
    col2.metric("Model confidence", f"{float(emotion.get('confidence') or 0):.1%}")
    st.caption(f"Reliability indicator: {emotion.get('reliability', '—')}")
    probabilities = emotion.get("probabilities") or {}
    for name in FINAL_CLASSES:
        value = min(1.0, max(0.0, float(probabilities.get(name, 0.0))))
        st.progress(value, text=f"{name.title()} · {value:.1%}")


def _render_video_result(result: dict[str, Any]) -> None:
    summary = result.get("summary")
    if not summary:
        st.warning(result.get("message") or "No temporal predictions were produced.")
        st.caption(
            f"Duration: {result.get('duration_seconds', 0):.1f}s · "
            f"Frames with a detected face: {result.get('sampled_frames_with_face', 0)}"
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Dominant expression", str(summary.get("dominant_expression", "—")).title())
    col2.metric("Average confidence", f"{float(summary.get('average_confidence') or 0):.1%}")
    col3.metric("Video duration", f"{float(result.get('duration_seconds') or 0):.1f}s")

    st.subheader("Average class probabilities")
    probabilities = summary.get("average_probabilities") or {}
    for name in FINAL_CLASSES:
        value = min(1.0, max(0.0, float(probabilities.get(name, 0.0))))
        st.progress(value, text=f"{name.title()} · {value:.1%}")

    timeline = result.get("timeline") or []
    if not timeline:
        st.info("The analysis completed but no ready predictions were available.")
        return

    st.subheader("Timeline")
    st.dataframe(timeline, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    fields = ["time", "label", "confidence", "reliability", *FINAL_CLASSES]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for row in timeline:
        probabilities_row = row.get("probabilities") or {}
        writer.writerow(
            {
                "time": row.get("time", ""),
                "label": row.get("label", ""),
                "confidence": row.get("confidence", ""),
                "reliability": row.get("reliability", ""),
                **{name: probabilities_row.get(name, "") for name in FINAL_CLASSES},
            }
        )

    st.download_button(
        "Download analysis as CSV",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="facialemotionai-analysis.csv",
        mime="text/csv",
        type="primary",
    )


def _run_video_analysis(uploaded_file: Any) -> None:
    suffix = Path(uploaded_file.name or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        st.error("Unsupported video format. Use MP4, MOV, AVI, MKV, WEBM, or M4V.")
        return

    payload = uploaded_file.getvalue()
    if not payload:
        st.error("The selected video is empty.")
        return
    if len(payload) > MAX_VIDEO_BYTES:
        st.error("The video exceeds the 100 MB limit.")
        return

    temp_path: str | None = None
    try:
        with st.spinner("Loading the AI models and analyzing the video…"):
            runtime = _session_runtime()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(payload)
                temp_path = temp_file.name
            result = _analyze_video_path(temp_path, runtime)
            st.session_state["facialemotionai_video_result"] = result
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        st.error(f"Video analysis failed: {detail or exc}")
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


st.markdown(
    """
    <div class="hero">
      <p class="muted">OPEN-SOURCE COMPUTER VISION · TEMPORAL MODEL</p>
      <h1>FacialEmotionAI</h1>
      <p class="muted">Webcam or video → face detection → Real/Fake screening → MobileNetV2 → 20-frame sequence → BiLSTM + Temporal Attention → five expression classes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Research demo only: the model estimates visible facial-expression categories. "
    "It does not determine a person's internal psychological state and is not a diagnostic tool."
)

mode = st.radio(
    "Choose an analysis mode",
    ["Upload a video", "Webcam"],
    horizontal=True,
    key="analysis_mode",
)

if mode == "Upload a video":
    st.subheader("Analyze a video")
    st.write("Upload a video up to 5 minutes long and 100 MB. Processing happens on the hosted app.")
    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        key="video_upload",
    )
    if st.button("Analyze video", type="primary", disabled=uploaded is None):
        _run_video_analysis(uploaded)

    previous_result = st.session_state.get("facialemotionai_video_result")
    if previous_result:
        _render_video_result(previous_result)

else:
    st.subheader("Live webcam analysis")
    st.write(
        "Allow camera access, then start the stream. The app samples frames to build "
        "the model's 20-frame temporal context."
    )

    if not st.session_state.get("facialemotionai_engine_ready"):
        if st.button("Load AI engine", type="primary"):
            try:
                with st.spinner("Loading TensorFlow models… this may take a little while on the free host."):
                    _session_runtime()
                st.session_state["facialemotionai_engine_ready"] = True
                st.success("AI engine loaded. Start the camera below.")
            except Exception as exc:
                st.session_state["facialemotionai_engine_ready"] = False
                st.error(f"Could not load the AI models: {exc}")

    if st.session_state.get("facialemotionai_engine_ready"):
        runtime = _session_runtime()
        context = webrtc_streamer(
            key="facialemotionai-webcam",
            video_processor_factory=lambda: EmotionVideoProcessor(runtime),
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            async_processing=True,
        )

        if context.state.playing and context.video_processor:
            live_results = st.empty()
            while context.state.playing:
                latest_result, latest_error = context.video_processor.get_latest()
                with live_results.container():
                    _render_live_result(latest_result, latest_error)
                time.sleep(0.4)

        if st.button("Reset temporal sequence"):
            runtime.reset()
            st.success("Temporal sequence reset. Restart the camera if it is running.")

st.divider()
st.caption(
    "Models are loaded only when requested. On free hosting, the first load may be slow or "
    "exceed available memory; public availability and inference speed are not guaranteed."
)
