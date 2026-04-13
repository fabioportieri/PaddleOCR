@echo off

REM Download file
powershell -Command "Invoke-WebRequest -Uri https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz -OutFile paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz"

REM Extract (requires tar available, Windows 10+)
tar -xf paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz

REM Copy config file
copy ..\pipeline_config_vllm.yaml paddlex_hps_PaddleOCR-VL-1.5_sdk\server\pipeline_config.yaml

echo Done.
pause