@echo off
title Cognivision AI - One-Click Launcher
color 0B

echo ============================================================
echo                COGNIVISION AI - LAUNCHER
echo ============================================================
echo.
echo Checking Python 3.13...
py -3.13 --version
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.13 was not found! Please ensure Python 3.13 is installed.
    pause
    exit /b %errorlevel%
)

echo.
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Cognivision AI - FastAPI Backend" cmd /k "py -3.13 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

echo.
echo [2/2] Starting React + Vite Frontend on http://localhost:5173 ...
start "Cognivision AI - React Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Waiting for servers to initialize...
timeout /t 3 /nobreak >nul

echo Opening Cognivision AI in your browser...
start http://localhost:5173

echo.
echo ============================================================
echo   Cognivision AI is now running!
echo   Frontend : http://localhost:5173
echo   Backend  : http://127.0.0.1:8000
echo   API Docs : http://127.0.0.1:8000/docs
echo ============================================================
echo Keep the server windows open while using the application.
echo To stop the servers, close their respective command windows.
echo.
pause
