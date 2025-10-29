@echo off
cd /d "%~dp0\.."
echo ========================================
echo TRD: Retrain with EOS Token Fix
echo ========================================
echo.
echo This retrains your model with the EOS token fix applied.
echo The fix ensures the model stops generating after responses.
echo.
echo Dataset: trd_framework\large_dataset\physics_large_dataset.jsonl
echo Output: trd_framework\models\qwen2-1.5b-1k-fixed
echo Time: ~35 minutes
echo.
echo Current directory: %CD%
echo.
echo ========================================
echo WHAT WAS FIXED:
echo ========================================
echo 1. Added EOS tokens to training data
echo 2. Fixed tokenizer configuration
echo 3. Model will now stop cleanly after responses
echo.
echo ========================================
echo STARTING TRAINING
echo ========================================
echo.

wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase2_training.py --dataset trd_framework/large_dataset/physics_large_dataset.jsonl --output-dir trd_framework/models/qwen2-1.5b-1k-fixed --batch-size 2 --gradient-accumulation 8 --epochs 3"

echo.
echo ========================================
echo TRAINING COMPLETE!
echo ========================================
echo.
echo Model saved to: trd_framework\models\qwen2-1.5b-1k-fixed
echo.
echo Next: Test the fixed model
echo   Run: trd_framework\TEST_FIXED_MODEL.bat
echo.
pause
