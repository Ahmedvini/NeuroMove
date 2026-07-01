FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system dependencies commonly needed by scientific Python stacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install coppeliasim-zmqremoteapi-client

COPY . /app

EXPOSE 8000

# Default to API service mode.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
