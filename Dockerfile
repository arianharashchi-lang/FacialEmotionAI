FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Git LFS is required because the trained Keras models are stored as LFS objects.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the application source. GitHub source archives may contain LFS pointer files,
# so fetch the real model artifacts from the public repository during the image build.
COPY . /app

RUN git lfs install --system \
    && git clone --depth 1 https://github.com/arianharashchi-lang/FacialEmotionAI.git /tmp/facialemotionai-models \
    && git -C /tmp/facialemotionai-models lfs pull \
    && mkdir -p /app/models \
    && cp -a /tmp/facialemotionai-models/models/. /app/models/ \
    && python - <<'PY'
from pathlib import Path

required = (
    "cnn_feature_extractor.keras",
    "fake_discriminator.keras",
    "lstm_attention_v1_best.keras",
    "sad_neutral_adapter.keras",
)
for name in required:
    path = Path("/app/models") / name
    if not path.is_file():
        raise SystemExit(f"Required model is missing: {path}")
    with path.open("rb") as f:
        header = f.read(128)
    if b"git-lfs.github.com/spec/v1" in header or path.stat().st_size < 100_000:
        raise SystemExit(f"{path} is an LFS pointer or unexpectedly small")
    print(f"Verified {path.name}: {path.stat().st_size:,} bytes")
PY

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-render.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
