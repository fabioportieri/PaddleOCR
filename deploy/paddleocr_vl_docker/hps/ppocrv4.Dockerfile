# ==============================
# PP-OCRv4 Standalone Service
# ==============================

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Upgrade pip
# ------------------------------
RUN pip install --no-cache-dir --upgrade pip

# ------------------------------
# Install PaddlePaddle (GPU version optional)
# ------------------------------
# If GPU:
# RUN pip install paddlepaddle-gpu==2.6.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# CPU fallback:
RUN pip install paddlepaddle==2.6.1

# ------------------------------
# Install PaddleOCR (v4 compatible)
# ------------------------------
RUN pip install paddleocr==2.7.0.3

# ------------------------------
# Create app directory
# ------------------------------
WORKDIR /app

# ------------------------------
# Download PP-OCRv4 models
# (You can also mount these later instead)
# ------------------------------
RUN mkdir -p /models/ppocrv4

# Optional: download models manually (recommended in production via volume)
# RUN wget -O det.tar https://.../PP-OCRv4_det_server_infer.tar && tar -xf det.tar -C /models/ppocrv4
# RUN wget -O rec.tar https://.../PP-OCRv4_rec_server_infer.tar && tar -xf rec.tar -C /models/ppocrv4
# RUN wget -O cls.tar https://.../cls_model.tar && tar -xf cls.tar -C /models/ppocrv4

# ------------------------------
# Copy service code
# ------------------------------
COPY ppocrv4/app.py /app/app.py

# ------------------------------
# Expose API port
# ------------------------------
EXPOSE 8082

# ------------------------------
# Run service
# ------------------------------
CMD ["python", "app.py"]