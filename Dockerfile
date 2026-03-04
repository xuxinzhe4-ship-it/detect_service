# Public Dockerfile template
# NOTE: This repository does NOT include best.onnx.
#       Provide your own weights at runtime (recommended) by mounting /app/best.onnx.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# If your base image already contains Python, you may remove python3/python3-pip below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py detector.py ./

EXPOSE 8000
CMD ["python3", "app.py"]
