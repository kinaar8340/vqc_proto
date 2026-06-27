FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ARG CACHEBUST=4
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import requests, gradio; print('deps ok', gradio.__version__)"

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

# HF Docker Space entrypoint
CMD python app.py