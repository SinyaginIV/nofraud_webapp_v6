@echo off
cd /d %~dp0

echo Активируем виртуальное окружение...
call venv\Scripts\activate

echo Запускаем backend...
start cmd /k "cd backend && python app.py"

timeout /t 2 >nul

echo Запускаем frontend на порту 8000...
start cmd /k "cd frontend && python -m http.server 8000"

echo Готово. Открой http://localhost:8000/index_v6.0.1.html
pause