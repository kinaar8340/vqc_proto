# VQC Proto — Orbital Braille quick demo + full pipeline
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt proto/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r proto/requirements.txt

COPY . .

# Default: Orbital Braille quick demo (seconds)
WORKDIR /app/proto
CMD ["python", "run_demo_quick.py", "--payload", "I live in Oregon", "--num-orbs", "4"]