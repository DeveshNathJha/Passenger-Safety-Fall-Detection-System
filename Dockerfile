# Inference server image: FastAPI + LiteRT (no TensorFlow) → small and fast to start.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FDS_LOG_FILE=""

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config/ config/
COPY models/*.tflite models/*.json models/
COPY src/ src/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
