# PowerShell script to create a Windows Scheduled Task for daily weather updates
# Run this script as Administrator

# Get the project root directory (one level up from scripts)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Configuration
$TaskName = "HRV-Lab Daily Weather Update"
$TaskDescription = "Fetches daily weather data for HRV-Lab from Open-Meteo API"
$BatchScriptPath = Join-Path $ProjectRoot "scripts\daily_weather_update.bat"
$TriggerTime = "6:00AM"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HRV-Lab Scheduled Task Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host "Batch Script: $BatchScriptPath" -ForegroundColor Yellow
Write-Host "Trigger Time: $TriggerTime daily" -ForegroundColor Yellow
Write-Host ""

# Check if running as Administrator
$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Please right-click and select 'Run as Administrator'" -ForegroundColor Red
    pause
    exit 1
}

# Check if batch script exists
if (-not (Test-Path $BatchScriptPath)) {
    Write-Host "ERROR: Batch script not found at: $BatchScriptPath" -ForegroundColor Red
    pause
    exit 1
}

# Remove existing task if it exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Existing task removed." -ForegroundColor Green
    Write-Host ""
}

# Create the action (what to run)
$Action = New-ScheduledTaskAction -Execute $BatchScriptPath -WorkingDirectory $ProjectRoot

# Create the trigger (when to run - daily at 6 AM)
$Trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

# Create the principal (run as current user)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Force | Out-Null

    Write-Host "SUCCESS: Scheduled task created!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Cyan
    Write-Host "  Name: $TaskName" -ForegroundColor White
    Write-Host "  Runs: Daily at $TriggerTime" -ForegroundColor White
    Write-Host "  Script: $BatchScriptPath" -ForegroundColor White
    Write-Host "  Logs: $ProjectRoot\logs\" -ForegroundColor White
    Write-Host ""
    Write-Host "You can manage this task in Task Scheduler or run it manually with:" -ForegroundColor Yellow
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host ""

    # Ask if user wants to run the task now
    $RunNow = Read-Host "Would you like to run the task now to test it? (y/n)"
    if ($RunNow -eq 'y' -or $RunNow -eq 'Y') {
        Write-Host ""
        Write-Host "Running task..." -ForegroundColor Yellow
        Start-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 2

        Write-Host "Task started. Check logs folder for output." -ForegroundColor Green
        Write-Host "Log location: $ProjectRoot\logs\" -ForegroundColor Cyan
    }

} catch {
    Write-Host "ERROR: Failed to create scheduled task" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
pause
