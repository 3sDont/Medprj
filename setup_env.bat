@echo off
echo ========================================
echo  MEDPRS - Virtual Environment Setup
echo ========================================

REM Create venv using the system Python (non-Store)
set PYTHON=C:\Users\Admin\AppData\Local\Python\bin\python3.exe

echo [1/3] Creating virtual environment (.venv)...
"%PYTHON%" -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Activating and installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [3/3] Done.
echo.
echo To activate the environment in this terminal, run:
echo     .venv\Scripts\activate.bat
echo.
echo VS Code: press Ctrl+Shift+P -^> "Python: Select Interpreter"
echo          and choose:  .\.venv\Scripts\python.exe
pause
