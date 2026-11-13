FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Set to "embeddings" to include sentence-transformers (large; keyword engine is the default).
ARG EXTRAS=""

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir ".[${EXTRAS}]"; else pip install --no-cache-dir .; fi

RUN useradd -m sentinel
USER sentinel

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8100/health')"
CMD ["sentinel-semantic", "--host", "0.0.0.0", "--port", "8100"]
