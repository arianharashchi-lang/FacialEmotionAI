FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt /app/requirements-render.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements-render.txt

COPY . /app

# The repository stores trained models in Git LFS. Download the real model
# artifacts during image build because source-build contexts can contain only
# the small LFS pointer files.
RUN git lfs install --system \
    && git clone --depth 1 --single-branch --branch main https://github.com/arianharashchi-lang/FacialEmotionAI.git /tmp/facialemotionai-models \
    && git -C /tmp/facialemotionai-models lfs pull \
    && mkdir -p /app/models \
    && cp -a /tmp/facialemotionai-models/models/. /app/models/ \
    && rm -rf /tmp/facialemotionai-models \
    && python -c "from pathlib import Path; names=('cnn_feature_extractor.keras','fake_discriminator.keras','lstm_attention_v1_best.keras','sad_neutral_adapter.keras'); missing=[n for n in names if not (Path('/app/models')/n).is_file() or (Path('/app/models')/n).stat().st_size < 100000]; assert not missing, f'Model files missing or still LFS pointers: {missing}'; print('Git LFS model artifacts verified')"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
