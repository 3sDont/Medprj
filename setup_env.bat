@echo off
echo ========================================
echo  MEDPRS - Virtual Environment Setup
echo ========================================

REM Prefer the py launcher (finds any installed Python) and fall back to
REM the known local install if it isn't on PATH.
set PYTHON=
where py >nul 2>nul
if not errorlevel 1 set PYTHON=py -3

if "%PYTHON%"=="" (
    if exist "C:\Python314\python.exe" (
        set PYTHON=C:\Python314\python.exe
    ) else (
        echo ERROR: Could not find a Python interpreter ^(tried "py" launcher and
        echo        C:\Python314\python.exe^). Install Python 3.10+ and re-run.
        pause
        exit /b 1
    )
)

echo [1/3] Creating virtual environment (.venv) with %PYTHON%...
%PYTHON% -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Activating and installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo [3/3] Done.
echo.
echo NOTE: requirements.txt installs CPU-only torch. For GPU support, install
echo       torch/torchvision matching your CUDA version FIRST from
echo       https://pytorch.org/get-started/locally/ then re-run this script
echo       (or just `pip install -r requirements.txt` again — it will skip
echo       torch if a compatible version is already installed).
echo.
echo To activate the environment in this terminal, run:
echo     .venv\Scripts\activate.bat
echo.
echo To run the app:
echo     streamlit run app.py
echo.
echo VS Code: press Ctrl+Shift+P -^> "Python: Select Interpreter"
echo          and choose:  .\.venv\Scripts\python.exe
pause
