FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# EXTRAS: "" (default, light, keyword engine) or e.g. "embeddings,classifier" (CPU torch + transformers; ~2 GB).
ARG EXTRAS=""
# CLASSIFIER_MODEL: Hugging Face model id to download at BUILD time so the container starts offline and fast.
ARG CLASSIFIER_MODEL=""
ENV HF_HOME=/home/sentinel/.cache/huggingface

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN if [ -n "$EXTRAS" ]; then \
      pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch && \
      pip install --no-cache-dir ".[${EXTRAS}]"; \
    else pip install --no-cache-dir .; fi

RUN useradd -m sentinel
USER sentinel
RUN if [ -n "$CLASSIFIER_MODEL" ]; then \
      python -c "import sys; from transformers import pipeline; pipeline('text-classification', model=sys.argv[1])" "$CLASSIFIER_MODEL"; \
    fi

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8100/health')"
CMD ["sentinel-semantic", "--host", "0.0.0.0", "--port", "8100"]
