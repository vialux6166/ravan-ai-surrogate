# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Default env
ENV UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Expose API port
EXPOSE 8000

# Default command: serve API
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
