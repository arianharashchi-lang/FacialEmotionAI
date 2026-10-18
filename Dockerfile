FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Git LFS is required because the trained Keras models are stored as LFS objects.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# GitHub source archives may contain LFS pointer files, so fetch the real models.
COPY . /app

RUN git lfs install --system \
    && git clone --depth 1 https://github.com/arianharashchi-lang/FacialEmotionAI.git /tmp/facialemotionai-models \
    && git -C /tmp/facialemotionai-models lfs pull \
    && mkdir -p /app/models \
    && cp -a /tmp/facialemotionai-models/models/. /app/models/ \
    && for name in cnn_feature_extractor.keras fake_discriminator.keras lstm_attention_v1_best.keras sad_neutral_adapter.keras; do \
         test -s "/app/models/$name" \
         && ! grep -aq "git-lfs.github.com/spec/v1" "/app/models/$name" \
         && size="$(stat -c%s "/app/models/$name")" \
         && test "$size" -ge 100000 \
         && echo "Verified $name ($size bytes)"; \
       done \
    && rm -rf /tmp/facialemotionai-models

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-render.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
