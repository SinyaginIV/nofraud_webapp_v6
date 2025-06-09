@echo off
cd /d %~dp0

call venv\Scripts\activate

start cmd /k "cd backend && python app.py"
timeout /t 2 >nul
start cmd /k "cd frontend && python -m http.server 8000"

echo Открой: http://localhost:8000/index_v6.0.1.html
pause