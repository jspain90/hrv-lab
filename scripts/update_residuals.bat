@echo off
REM Update HRV residuals using latest model
REM Run this after new HRV data is ingested

setlocal

REM Get project root
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM Check if model exists
if not exist "models\hrv_baseline.pkl" (
    echo ERROR: Model not found at models\hrv_baseline.pkl
    echo Please train a model first: python pipeline\train_baseline_model.py --save-model models\hrv_baseline.pkl
    pause
    exit /b 1
)

echo Updating HRV residuals...
echo.

REM Compute residuals for last 30 days
python pipeline\compute_residuals.py --model models\hrv_baseline.pkl

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Residuals updated successfully
) else (
    echo.
    echo ERROR: Residuals update failed
)

echo.
pause
