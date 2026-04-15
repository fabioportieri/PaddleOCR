FROM paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python deps
RUN python -m pip install --upgrade pip

# Core
RUN pip install paddleocr

# REQUIRED for PP-StructureV3
RUN pip install "paddlex[ocr]"

# install  api stack
RUN pip install fastapi uvicorn python-multipart opencv-python

# optional utils
RUN pip install numpy pillow

# copy service
COPY sources/ppstructure_service.py /app/ppstructure_service.py

EXPOSE 8082

CMD ["uvicorn", "ppstructure_service:app", "--host", "0.0.0.0", "--port", "8082"]