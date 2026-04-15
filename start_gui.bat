@echo off
REM start_gui.bat
REM Quick launcher for Flux Training GUI

echo ========================================
echo   Flux Training GUI
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup_all.ps1 first.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if main.py exists
if not exist "main.py" (
    echo Error: main.py not found!
    echo Make sure you're in the correct directory.
    echo.
    pause
    exit /b 1
)

echo Starting GUI...
echo.

REM Launch the GUI
python main.py

REM Deactivate when done
call .venv\Scripts\deactivate.bat

echo.
echo GUI closed.
pause
