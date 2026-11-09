@echo off
title Passenger Safety ^& Fall Detection System
REM ------------------------------------------------------------------
REM  Starts the API for a phone on the same Wi-Fi.
REM  Phones only expose motion sensors over HTTPS, so if key.pem /
REM  cert.pem exist (e.g. created with mkcert) the server uses TLS.
REM  Set VENV to your virtual environment folder (default: .venv).
REM ------------------------------------------------------------------
if "%VENV%"=="" if exist "C:\fall_detection_env\Scripts\python.exe" set VENV=C:\fall_detection_env
if "%VENV%"=="" if exist "fall_detection_env\Scripts\python.exe" set VENV=fall_detection_env
if "%VENV%"=="" set VENV=.venv

for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.PrefixOrigin -ne 'WellKnown'} | Select-Object -First 1).IPAddress"`) do set IP=%%i
if "%IP%"=="" set IP=127.0.0.1

set SSL_ARGS=
set PROTO=http
if exist key.pem if exist cert.pem (
    set SSL_ARGS=--ssl-keyfile key.pem --ssl-certfile cert.pem
    set PROTO=https
)

echo ============================================================
echo   Open on your phone:  %PROTO%://%IP%:8000/demo
echo   API docs:            %PROTO%://%IP%:8000/docs
echo ============================================================
if "%PROTO%"=="http" echo   [!] No key.pem/cert.pem found - most phone browsers block motion sensors over plain HTTP.

"%VENV%\Scripts\python.exe" -m uvicorn src.app:app --host 0.0.0.0 --port 8000 %SSL_ARGS%
pause
