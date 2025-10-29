@echo off
cd /d "%~dp0\.."
echo ========================================
echo TRD: Generate 1,000 Examples (Windows)
echo ========================================
echo.
echo This runs in Windows (not WSL) to avoid networking issues.
echo.
echo Teacher Model: qwen3:30b-a3b
echo Target: 1,000 examples
echo Time: ~3 hours
echo.
echo Current directory: %CD%
echo.
echo ========================================
echo STEP 1: Starting Ollama
echo ========================================
echo.
start "Ollama - qwen3:30b-a3b" cmd /k "ollama run qwen3:30b-a3b"
echo Waiting 15 seconds for model to load...
timeout /t 15 /nobreak
echo.
echo ========================================
echo STEP 2: Generating Dataset
echo ========================================
echo.
echo Running in Windows Python (not WSL)...
echo.

python trd_framework\generate_dataset_windows.py --target-size 1000 --teacher-model qwen3:30b-a3b --output-dir trd_framework\large_dataset

echo.
echo ========================================
echo Complete!
echo ========================================
echo.
pause
