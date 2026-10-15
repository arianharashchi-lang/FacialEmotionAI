# FacialEmotionAI FastAPI + ngrok MVP

This folder exposes the existing local FacialEmotionAI model stack through a lightweight FastAPI web interface.

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

## 1. Install the web dependencies

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r webapp\requirements.txt
```

## 2. Start FastAPI

```powershell
uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Local address:

```text
http://127.0.0.1:8000
```

Keep this terminal open.

## 3. Install ngrok on Windows

Recommended Windows command:

```powershell
winget install ngrok -s msstore
```

Create a free ngrok account and copy the account authtoken, then run:

```powershell
ngrok config add-authtoken YOUR_AUTHTOKEN
```

## 4. Publish the local FastAPI app

In a second PowerShell window:

```powershell
ngrok http 8000
```

Open the HTTPS ngrok address from a phone, tablet, or another computer.

Free ngrok accounts receive an assigned development domain. The local laptop still performs TensorFlow/OpenCV inference, so FastAPI and ngrok must both stay running.

## Scope

FacialEmotionAI estimates visible facial-expression categories. It does not infer internal psychological state and is not a clinical, hiring, lie-detection, or diagnostic system.

## MVP concurrency

The current web runtime intentionally shares one loaded TensorFlow model stack. Treat this version as a one-active-analysis-session demo. Multi-user production deployment would require isolated session state and dedicated inference workers.
