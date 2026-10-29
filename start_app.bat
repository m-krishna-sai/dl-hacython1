@echo off
echo ===================================================
echo           Starting TRUSTMED Application
echo ===================================================
echo.

cd /d C:\Users\kodur\.gemini\antigravity\scratch\dl-hacython1

echo [1/2] Launching Backend FastAPI Server in new window...
start "TRUSTMED Backend (FastAPI)" cmd /k "cd /d C:\Users\kodur\.gemini\antigravity\scratch\dl-hacython1 && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Launching Frontend (React + Vite) in new window...
start "TRUSTMED Frontend (Vite)" cmd /k "cd /d C:\Users\kodur\.gemini\antigravity\scratch\dl-hacython1\frontend && npm run dev"

echo.
echo Both servers launched successfully!
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo.
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://localhost:5173
echo Done!
pause
