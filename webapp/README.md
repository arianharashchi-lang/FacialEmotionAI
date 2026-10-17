# FacialEmotionAI FastAPI Web Demo

This folder exposes the existing FacialEmotionAI model stack through a web interface.

## Features

- Responsive desktop/mobile web UI
- Front/back camera selection on supported phones
- 20-frame temporal webcam analysis
- Existing Real/Fake screening gate
- Existing five-class temporal model
- Video upload analysis
- Timeline output
- CSV export
- Print / Save as PDF reporting

## Deploy on Render

The repository includes a root-level `render.yaml` Blueprint and a build script.

1. Open Render and sign in with GitHub.
2. Choose **New + → Blueprint**.
3. Connect `https://github.com/arianharashchi-lang/FacialEmotionAI`.
4. Select branch `main` and let Render read `render.yaml`.
5. Review the service settings and create the Blueprint.
6. Watch the build and deploy logs. The build script installs Git LFS, downloads the model artifacts, verifies that they are real model files rather than LFS pointer text, and installs the web dependencies.
7. Open the generated `onrender.com` URL and test `/health`, then click **Load AI Engine**.

The Blueprint uses Render's free plan. Free instances have limited CPU/RAM and sleep after inactivity. TensorFlow model loading or inference can exceed free-instance resources; if logs show an out-of-memory termination, this app will need a higher-memory host or a more lightweight inference setup.

The web runtime shares one model stack and is intended as a research/portfolio demo, not a multi-user production service. Uploaded videos are processed temporarily and deleted after analysis. Free Render storage is ephemeral, so do not use the service to store persistent reports or databases.

## Local run

Install dependencies from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r webapp\requirements.txt
```

Start FastAPI:

```powershell
uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` and keep the terminal running.

## Scope

FacialEmotionAI estimates visible facial-expression categories. It does not infer internal psychological state and is not a clinical, hiring, lie-detection, or diagnostic system.
