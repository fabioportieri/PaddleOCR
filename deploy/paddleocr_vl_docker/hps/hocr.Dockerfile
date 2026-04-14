FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY hocr_api ./hocr_api
COPY sources ./sources

RUN python -m pip install --no-cache-dir -r hocr_api/requirements.txt

ENV HPS_LAYOUT_PARSING_URL=http://paddleocr-vl-api:8080/layout-parsing
ENV HPS_LAYOUT_PARSING_TIMEOUT=600
ENV HPS_LOG_LEVEL=INFO
ENV UVICORN_WORKERS=1

EXPOSE 8081

CMD uvicorn --host 0.0.0.0 --port 8081 --workers ${UVICORN_WORKERS} hocr_api.app:app
