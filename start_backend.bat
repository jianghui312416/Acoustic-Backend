@echo off
setlocal
title Acoustic Field GeoVis - Backend
cd /d "%~dp0"

echo [Backend] Project: %CD%

if not exist "backworker\backend_worker.py" (
    echo [ERROR] backworker\backend_worker.py was not found in %CD%.
    pause
    exit /b 1
)

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

rem 方位级并行由应用控制。禁止 NumPy/BLAS 在每个方位线程内再次扩张线程，
rem 避免 4 进程 x 4 线程被底层库放大成数十个线程。
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"

if not exist "%PYTHON_EXE%" (
    echo [Backend] Python virtual environment is missing. Creating .venv...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv ".venv"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Python 3 is not installed or is not available in PATH.
            pause
            exit /b 1
        )
        python -m venv ".venv"
    )
    if errorlevel 1 (
        echo [ERROR] Failed to create the Python virtual environment.
        pause
        exit /b 1
    )
)

"%PYTHON_EXE%" -c "import fastapi, uvicorn, numpy, scipy, h5py, netCDF4, requests, websockets, psycopg2" >nul 2>&1
if errorlevel 1 (
    echo [Backend] Dependencies are missing. Installing requirements.txt...
    "%PYTHON_EXE%" -m pip install -r "requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo [Backend] Starting http://127.0.0.1:8888 ...
echo [Backend] API documentation: http://127.0.0.1:8888/docs
echo [Backend] Press Ctrl+C to stop.
"%PYTHON_EXE%" -m uvicorn backworker.backend_worker:app --host 0.0.0.0 --port 8888
set "BACKEND_EXIT_CODE=%ERRORLEVEL%"

if not "%BACKEND_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Backend stopped unexpectedly with exit code %BACKEND_EXIT_CODE%.
    echo Check the messages above, Windows Event Viewer, and whether port 8888 is already in use.
    pause
    exit /b %BACKEND_EXIT_CODE%
)

endlocal
