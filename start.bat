@echo off
title Offline Curriculum AI Assistant
echo ===================================================
echo   Starting Offline Curriculum AI Assistant...
echo ===================================================

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11 or 3.12 from python.org.
    pause
    exit /b
)

:: 2. Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [1/4] Creating isolated Python virtual environment...
    python -m venv venv
)

:: 3. Activate virtual environment and install dependencies
call venv\Scripts\activate.bat
echo [2/4] Verifying and installing dependencies...
pip install -q -r requirements.txt

:: 4. Check Ollama and pull models if missing
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama not found! Please install it from https://ollama.com
    pause
    exit /b
)

echo [3/4] Checking local AI models...
ollama pull llama3.2
ollama pull nomic-embed-text

:: 5. Launch Streamlit App
echo [4/4] Launching web interface...
streamlit run app.py
pause