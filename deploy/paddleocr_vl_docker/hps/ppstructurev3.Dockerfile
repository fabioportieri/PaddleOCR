  # Use the official NVIDIA CUDA 11.8 devel image for Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Install system dependencies for OpenCV, Paddle, and PDF processing
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set official Paddle mirror for the framework
RUN pip install --upgrade pip

# Install the latest stable Paddle Framework for CUDA 11.8
RUN pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install PaddleOCR and PaddleX with all plugins
# Pinning numpy < 2.0 ensures compatibility with the current Paddle runtime
RUN pip install "numpy<2.0"
RUN pip install "paddleocr>=3.4.1" "paddlex[ocr]>=3.4.3"
RUN pip install "uvicorn[standard]" fastapi python-multipart attrs

# Disable model source check to speed up startup in production
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Create app directory and set as working directory
RUN mkdir -p /app
WORKDIR /app

# Copy sources
COPY sources/ /app/


EXPOSE 8082

CMD ["uvicorn", "ppstructure_service:app", "--host", "0.0.0.0", "--port", "8082"]