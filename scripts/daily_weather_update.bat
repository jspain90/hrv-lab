@echo off
REM Daily Weather Data Update Script
REM Fetches weather data for the last 7 days to ensure no gaps

setlocal

REM Get project root directory (one level up from scripts)
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Generate log filename with date
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
set LOGFILE=logs\weather_update_%mydate%.log

echo ========================================== >> "%LOGFILE%"
echo Weather Data Update >> "%LOGFILE%"
echo Started: %date% %time% >> "%LOGFILE%"
echo ========================================== >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM Run the weather fetch script
REM Using --days 7 to fetch last week's data (handles any gaps)
python pipeline\weather_fetch.py --days 7 >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo. >> "%LOGFILE%"
    echo SUCCESS: Weather data updated successfully >> "%LOGFILE%"
) else (
    echo. >> "%LOGFILE%"
    echo ERROR: Weather update failed with code %ERRORLEVEL% >> "%LOGFILE%"
)

echo. >> "%LOGFILE%"
echo Finished: %date% %time% >> "%LOGFILE%"
echo ========================================== >> "%LOGFILE%"
echo. >> "%LOGFILE%"

endlocal
