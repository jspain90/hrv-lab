@echo off
REM Quick script to view the most recent weather update log

setlocal

REM Get project root
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM Find the most recent log file
for /f "delims=" %%i in ('dir /b /o-d logs\weather_update_*.log 2^>nul') do (
    echo Showing latest log: logs\%%i
    echo.
    type "logs\%%i"
    goto :done
)

echo No log files found in logs directory.
echo Run the weather update script first: scripts\daily_weather_update.bat

:done
echo.
pause
