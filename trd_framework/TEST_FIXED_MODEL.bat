@echo off
cd /d "%~dp0\.."
echo ========================================
echo TRD: Test Fixed Model
echo ========================================
echo.
echo This tests your retrained model with proper EOS token handling.
echo.
echo Model: trd_framework\models\qwen2-1.5b-1k-fixed
echo.
echo ========================================
echo Choose Test Mode:
echo ========================================
echo 1. Examples Mode (3 physics questions)
echo 2. Interactive Mode (ask your own questions)
echo.
set /p choice="Enter choice (1 or 2): "
echo.

if "%choice%"=="1" (
    echo Running Examples Mode...
    echo.
    wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/test_trd_model.py --model-path trd_framework/models/qwen2-1.5b-1k-fixed --mode examples --domain physics"
) else if "%choice%"=="2" (
    echo Running Interactive Mode...
    echo Type your questions and press Enter.
    echo Type 'quit' to exit.
    echo.
    wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/test_trd_model.py --model-path trd_framework/models/qwen2-1.5b-1k-fixed --mode interactive"
) else (
    echo Invalid choice. Please run again and choose 1 or 2.
)

echo.
pause
