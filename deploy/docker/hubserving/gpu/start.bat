@echo off
setlocal

cd /d "%~dp0"
docker compose -f compose.yaml up -d --build

endlocal
