from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import av
import cv2
import pandas as pd
import streamlit as st
from fastapi import HTTPException
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from webapp.app import _analyze_video_path, get_runtime


MAX_VIDEO_BYTES = 100 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}
EMOTION_CLASSES = ["angry", "happy", "surprise", "sad", "neutral"]


st.set_page_config(
    page_title="FacialEmotionAI | Public Demo",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(42, 94, 166, .20), transparent 35%),
            #07111f;
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .82); }
    .hero {
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(116, 167, 255, .25);
        border-radius: 20px;
        background: linear-gradient(125deg, rgba(20, 44, 75, .92), rgba(11, 24, 42, .92));
        margin-bottom: 1.2rem;
    }
    .hero-eyebrow {
        color: #8bbcff;
        font-size: .78rem;
        letter-spacing: .16em;
        font-weight: 700;
    }
    .hero h1 { margin: .35rem 0 .4rem 0; color: #f4f8ff; }
    .hero p { color: #b9c9df; margin: 0; }
    div[data-testid="stMetric"] {
        background: rgba(15, 31, 51, .78);
        border: 1px solid rgba(116, 167, 255, .18);
        padding: 1rem;
        border-radius: 14px;
    }
    </style>
    <div class="hero">
      <div class="hero-eyebrow">FACIALEMOTIONAI · RESEARCH DEMO</div>
      <h1>Facial-expression analysis</h1>
      <p>Webcam video or uploaded clip · Real/Fake screening · 20-frame temporal model · Five visible-expression classes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "This demo estimates visible facial-expression categories. It does not determine "
    "a person's internal emotional state and is not a clinical or diagnostic tool."
)


class FacialEmotionVideoProcessor(VideoProcessorBase):
    """Draw lightweight live predictions over a browser webcam stream."""

    def __init__(self) -> None:
        self.runtime = get_runtime()
        self.runtime.reset()
        self.last_processed_at = 0.0
        self.latest_result: dict[str, Any] = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()

        # Limit model work to roughly 8 frames per second.
        if now - self.last_processed_at >= 0.12:
            try:
                self.latest_result = self.runtime.analyze(image)
            except Exception as exc:
                self.latest_result = {"error": type(exc).__name__}
            self.last_processed_at = now

        result = self.latest_result
        color = (92, 190, 255)
        label = "Looking for a face..."

        if result.get("error"):
            label = "Analysis temporarily unavailable"
            color = (70, 70, 235)
        elif result.get("face_detected"):
            box = result.get("face_box")
            if box and len(box) == 4:
                x, y, width, height = [int(value) for value in box]
                cv2.rectangle(image, (x, y), (x + width, y + height), (70, 220, 160), 2)

            anti_spoof = result.get("anti_spoof") or {}
            emotion = result.get("emotion") or {}

            if anti_spoof.get("status") == "FAKE":
                label = "Frame rejected by Real/Fake gate"
                color = (70, 70, 235)
            elif emotion.get("ready"):
                label = (
                    f"{str(emotion.get('label', 'unknown')).upper()}  "
                    f"{float(emotion.get('confidence') or 0.0):.0%}"
                )
                color = (70, 220, 160)
            else:
                label = (
                    f"Temporal warm-up: {emotion.get('collected', 0)}"
                    f"/{emotion.get('required', 20)}"
                )

        cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 440), 48), (7, 17, 31), -1)
        cv2.putText(
            image,
            label[:58],
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def timeline_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for point in result.get("timeline", []):
        row: dict[str, Any] = {
            "Time (seconds)": point.get("time"),
            "Expression": str(point.get("label") or "").title(),
            "Confidence (%)": round(float(point.get("confidence") or 0.0) * 100, 2),
            "Reliability": point.get("reliability") or "",
        }
        probabilities = point.get("probabilities") or {}
        for name in EMOTION_CLASSES:
            row[f"{name.title()} (%)"] = round(float(probabilities.get(name) or 0.0) * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def render_video_result(result: dict[str, Any]) -> None:
    summary = result.get("summary")
    if not summary:
        st.warning(result.get("message") or "No temporal predictions were produced.")
        st.caption(
            "Try a clearer, front-facing clip with one visible face and enough frames "
            "for the 20-frame temporal buffer."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Dominant expression", str(summary.get("dominant_expression", "—")).title())
    col2.metric("Average confidence", f"{float(summary.get('average_confidence') or 0.0):.1%}")
    col3.metric("Video duration", f"{float(result.get('duration_seconds') or 0.0):.1f} s")

    probabilities = summary.get("average_probabilities") or {}
    chart_data = pd.DataFrame(
        {
            "Average probability (%)": [
                round(float(probabilities.get(name) or 0.0) * 100, 2)
                for name in EMOTION_CLASSES
            ]
        },
        index=[name.title() for name in EMOTION_CLASSES],
    )
    st.subheader("Average class probabilities")
    st.bar_chart(chart_data)

    timeline = timeline_dataframe(result)
    st.subheader("Prediction timeline")
    if timeline.empty:
        st.info("The video was processed, but no ready predictions were returned.")
    else:
        st.dataframe(timeline, use_container_width=True, hide_index=True)
        csv_bytes = timeline.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download timeline as CSV",
            data=csv_bytes,
            file_name="facialemotionai-timeline.csv",
            mime="text/csv",
            key="download-timeline-csv",
        )

    st.download_button(
        "Download full result as JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="facialemotionai-result.json",
        mime="application/json",
        key="download-full-result-json",
    )


with st.sidebar:
    st.header("AI engine")
    st.write("Load the model stack once before using the webcam or video analyzer.")
    if st.button("Load AI engine", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading TensorFlow models. The first load may take a while..."):
                get_runtime()
            st.session_state["engine_loaded"] = True
            st.success("AI engine is ready.")
        except Exception as exc:
            st.session_state["engine_loaded"] = False
            st.error(f"Could not load the AI engine: {type(exc).__name__}: {exc}")

    if st.session_state.get("engine_loaded"):
        st.success("Engine loaded")
    else:
        st.info("Engine not loaded yet")

    st.divider()
    st.caption("Limits: video up to 100 MB and 5 minutes.")
    st.caption("For reliable results, use one active analysis at a time.")


if not st.session_state.get("engine_loaded"):
    st.info("Start by clicking **Load AI engine** in the left sidebar.")
else:
    live_tab, upload_tab = st.tabs(["Live webcam", "Upload / record a video"])

    with live_tab:
        st.subheader("Live webcam analysis")
        st.write(
            "Allow camera access, then press START. The overlay shows face detection, "
            "Real/Fake screening, temporal warm-up, and the current expression."
        )
        webrtc_streamer(
            key="facialemotionai-live-webcam",
            video_processor_factory=FacialEmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTC_CONFIGURATION,
            async_processing=True,
        )
        st.caption(
            "If your network blocks direct WebRTC connections, use the video tab to "
            "record a webcam clip or upload an existing clip."
        )

    with upload_tab:
        st.subheader("Analyze a video")
        uploaded_video = st.file_uploader(
            "Choose a video or record one from your webcam",
            type=sorted(ALLOWED_VIDEO_SUFFIXES - {".m4v"}),
            help="Maximum 100 MB and 5 minutes. Videos are processed temporarily and not saved as a user library.",
        )

        if st.button(
            "Analyze video",
            type="primary",
            disabled=uploaded_video is None,
            use_container_width=False,
        ):
            suffix = Path(uploaded_video.name or "video.mp4").suffix.lower()
            if suffix not in ALLOWED_VIDEO_SUFFIXES:
                st.error("Unsupported video format.")
            elif uploaded_video.size > MAX_VIDEO_BYTES:
                st.error("Video exceeds the 100 MB limit.")
            else:
                temporary_path: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                        temporary_path = temporary_file.name
                        temporary_file.write(uploaded_video.getbuffer())

                    with st.spinner("Analyzing video. Keep this page open..."):
                        result = _analyze_video_path(
                            temporary_path,
                            runtime_override=get_runtime(),
                        )
                    st.session_state["latest_video_result"] = result
                    st.success(result.get("message") or "Analysis complete.")
                except HTTPException as exc:
                    st.error(str(exc.detail))
                except Exception as exc:
                    st.error(f"Video analysis failed: {type(exc).__name__}: {exc}")
                finally:
                    if temporary_path:
                        try:
                            os.remove(temporary_path)
                        except OSError:
                            pass

        result = st.session_state.get("latest_video_result")
        if result:
            st.divider()
            render_video_result(result)


st.divider()
st.caption(
    "Research/portfolio demo only. The model was trained on limited acted-expression data; "
    "lighting, pose, occlusion, and individual variation can affect results."
)
