# Cognivision AI PowerShell Launcher

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                COGNIVISION AI - LAUNCHER                   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verify Python 3.13
$pyVersion = py -3.13 --version 2>&1
Write-Host "Detected: $pyVersion" -ForegroundColor Green
Write-Host ""

# Start Backend in separate PowerShell window
Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "py -3.13 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

# Start Frontend in separate PowerShell window
Write-Host "[2/2] Starting React + Vite Frontend on http://localhost:5173 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

# Wait 3 seconds and launch default browser
Start-Sleep -Seconds 3
Write-Host "Opening Cognivision AI in your browser..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Cognivision AI is now live!                              " -ForegroundColor Green
Write-Host "   Frontend : http://localhost:5173                         " -ForegroundColor White
Write-Host "   Backend  : http://127.0.0.1:8000                         " -ForegroundColor White
Write-Host "   API Docs : http://127.0.0.1:8000/docs                    " -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Close the spawned server windows to shut down the app." -ForegroundColor Gray
