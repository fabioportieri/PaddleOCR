FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# CRITICAL: Use the official Paddle mirror for the GPU wheel
RUN pip install paddlepaddle-gpu==2.6.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install OCR tools - Pin numpy to avoid compatibility crashes with Paddle
RUN pip install "numpy<2.0"
RUN pip install paddleocr==2.8.1 "paddlex[ocr]==3.0.0"

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# copy service
COPY sources/ppstructure_service.py /app/ppstructure_service.py

EXPOSE 8082

CMD ["uvicorn", "ppstructure_service:app", "--host", "0.0.0.0", "--port", "8082"]