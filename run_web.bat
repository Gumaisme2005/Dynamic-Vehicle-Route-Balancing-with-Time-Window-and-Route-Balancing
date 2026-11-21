@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Khong tim thay .venv\Scripts\python.exe
    pause
    exit /b 1
)
echo Dang khoi dong Web Dashboard tai http://127.0.0.1:8000 ...
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m web.server --host 127.0.0.1 --port 8000
