@echo off
title Passenger Safety & Fall Detection System - Starter
echo ============================================================
echo   Passenger Safety & Fall Detection System (Windows Run Script)
echo ============================================================
echo.
echo [1/3] Finding your Wi-Fi/Local IP address...

for /f "usebackq tokens=*" %%i in (`powershell -Command "$ip = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi' -ErrorAction SilentlyContinue).IPAddress; if (-not $ip) { $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*'} | Select-Object -First 1).IPAddress }; $ip"`) do set IP=%%i

if "%IP%"=="" (
    echo [!] Could not automatically detect local IP. 
    echo Please make sure you are connected to Wi-Fi.
    echo Defaulting to localhost.
    set IP=127.0.0.1
)

echo.
echo ============================================================
echo   👉 Open this link on your Mobile Phone:
echo   https://%IP%:8000/demo
echo ============================================================
echo.
echo   *Note: Both laptop and phone must be on the SAME Wi-Fi network.*
echo   *Note: Tap "Advanced" -> "Proceed" when browser warning appears.*
echo.
echo ============================================================
echo [2/3] Starting FastAPI (Uvicorn) with HTTPS (SSL)...
echo ============================================================
echo.

.\fall\Scripts\uvicorn.exe src.app:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem --reload

echo.
echo [3/3] Server stopped.
pause
