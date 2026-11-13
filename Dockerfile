# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn pydantic sentence-transformers pyyaml pytest httpx

# Copy source code, configuration, and tests
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Set Python path so modules resolve correctly
ENV PYTHONPATH=/app/src

# Expose port for FastAPI gateway
EXPOSE 8000

# Default command to run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]