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

## Deploy on Google Cloud Run

The recommended hosted deployment uses the existing FastAPI app in a Docker container. It does not use Streamlit and does not require a custom domain.

Follow the repository guide: [Deploy on Google Cloud Run](../DEPLOY_GOOGLE_CLOUD_RUN.md).

**Billing note:** Cloud Run has a monthly free usage allowance, but it is not guaranteed to remain at zero cost. A billing account is required, and usage beyond the free quotas or image/build storage can be charged. The guide limits the service to one instance and recommends setting a budget alert before deploying.

The Docker build installs CPU-only TensorFlow and headless OpenCV, downloads the real model artifacts from Git LFS, and verifies them before starting Uvicorn. After deployment, test `/health` and then click **Load AI Engine** on the homepage.

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
