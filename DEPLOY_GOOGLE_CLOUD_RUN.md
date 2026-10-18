# Deploy FacialEmotionAI on Google Cloud Run

This deploys the existing FastAPI web app (not Streamlit) with an HTTPS `run.app` URL. The container build downloads the trained model files from the repository's Git LFS storage.

## Important cost note

Cloud Run has a monthly free usage allowance, but this is **not a guarantee of zero cost**. A Google Cloud billing account is required, and usage beyond free quotas, container image storage, or build usage may be billed. Set a budget alert before deploying. A budget alert is a notification, not a hard spending cap. The service is configured below for one instance to reduce unexpected scaling.

## Prerequisites

- A Google account and a Google Cloud project.
- Billing enabled for that project.
- Google Cloud Shell (available from the Google Cloud Console toolbar).

No custom domain is required. Cloud Run provides an HTTPS URL, which is important for browser camera access.

## Steps

1. Open the [Google Cloud Console](https://console.cloud.google.com/), create or select a project, and make sure billing is enabled.
2. Open **Cloud Shell** from the top-right terminal icon.
3. Replace `YOUR_PROJECT_ID` with the project ID shown in the console and run:

   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

4. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/arianharashchi-lang/FacialEmotionAI.git
   cd FacialEmotionAI
   ```

5. Deploy the service:

   ```bash
   gcloud run deploy facialemotionai-web \
     --source . \
     --region europe-west3 \
     --allow-unauthenticated \
     --port 8080 \
     --memory 4Gi \
     --cpu 1 \
     --max 1 \
     --concurrency 1 \
     --timeout 900s
   ```

   When prompted to create an Artifact Registry repository, answer `Y`.

6. Wait for the build and deployment to finish. The command prints the public service URL. Open it in a browser and check that `/health` returns JSON containing `"status":"ok"`. Then open the homepage and click **Load AI Engine**.

## Notes

- The image uses Python 3.11, CPU-only TensorFlow, and headless OpenCV.
- Model weights are stored with Git LFS; the Docker build downloads and verifies the actual artifacts.
- The web app loads TensorFlow and its models only when **Load AI Engine** is clicked, so the health endpoint and homepage can start before the AI engine is loaded.
- Uploaded videos are temporary and are deleted after analysis. Cloud Run's writable filesystem is ephemeral; do not rely on local SQLite logs or uploaded files surviving instance restarts.
- This is a research/portfolio demo, not a clinical or psychological diagnostic system.
