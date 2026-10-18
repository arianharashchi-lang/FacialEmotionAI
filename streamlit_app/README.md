# Streamlit Community Cloud demo

This app reuses the existing FacialEmotionAI inference runtime and model artifacts.

## Deploy

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Create an app from this repository.
3. Branch: `main`.
4. Entrypoint: `streamlit_app/app.py`.
5. In Advanced settings, select Python 3.11 if available, then deploy.
6. The app uses the repository's Git LFS model artifacts; Streamlit Community Cloud supports Git LFS.

## Features

- Browser webcam analysis through WebRTC
- Video upload analysis (MP4, MOV, AVI, MKV, WEBM, M4V; maximum five minutes and 100 MB)
- Five-class temporal output, confidence/reliability indicators, timeline table, and CSV export

The models are loaded on demand to avoid loading TensorFlow during initial page startup. Free-host resource limits can still prevent model loading or inference; if that happens, inspect the Streamlit app logs before changing code.

This is a research/portfolio demo, not a clinical or psychological diagnostic system.
