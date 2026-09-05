FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EVIDENCE_AI_ENABLED=false \
    MODEL_EXPLANATION_AI_ENABLED=false \
    SUPPORT_SIGNAL_AI_ENABLED=false \
    RAZORPAY_INTEGRATION_ENABLED=false \
    RAZORPAY_TEST_MODE=true \
    RAZORPAY_WEBHOOK_ENABLED=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# Binary model artifacts and generated data are intentionally not committed.
# Recreate the canonical baseline during the immutable image build.
RUN python -m src.pipeline

RUN groupadd --system radar \
    && useradd --system --gid radar --home-dir /app radar \
    && mkdir -p /app/runtime \
    && chown radar:radar /app/runtime

USER radar

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
