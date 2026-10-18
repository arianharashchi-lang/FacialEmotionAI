# Streamlit Community Cloud demo

This folder contains an alternate public interface for FacialEmotionAI. It reuses the existing TensorFlow/OpenCV runtime and supports:

- Browser webcam analysis with a temporal overlay
- Uploading or recording a video in the browser
- Five-class summary and prediction timeline
- CSV and JSON result exports

## Deploy for free

1. Make sure the streamlit_demo files are on the repository's main branch.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Click Create app and select Yup, I have an app.
4. Select repository arianharashchi-lang/FacialEmotionAI and branch main.
5. Set Main file path to streamlit_demo/streamlit_app.py.
6. In Advanced settings, select Python 3.11.
7. Deploy the app. Streamlit Community Cloud will install the dependencies from streamlit_demo/requirements.txt; Git LFS model files are supported by Community Cloud.

The app will receive a free streamlit.app URL. Free apps can sleep after inactivity and have resource limits; the first wake-up/model load may take time. This is a research/portfolio demo, not a multi-user production service.

## Local test

Run from the repository root so imports resolve consistently:

    python -m pip install -r streamlit_demo/requirements.txt
    streamlit run streamlit_demo/streamlit_app.py

If live webcam video cannot connect from a particular network, use the Upload / record a video tab. Some network environments require a TURN relay for WebRTC.
