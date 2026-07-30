@echo off
REM ============================================
REM  AI Toolkit Modal - Start UI
REM ============================================

echo.
echo  Checking prerequisites...
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

REM Check Modal CLI
modal --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Modal CLI is not installed.
    echo         Install it with: pip install modal
    echo         Then authenticate with: modal setup
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('modal --version 2^>^&1') do set MODALVER=%%v
echo [OK] Modal CLI %MODALVER% found.

echo.
echo  Starting AI Toolkit UI on Modal...
echo  Press Ctrl+C to stop.
echo.

modal serve run_ai_toolkit_ui.py
