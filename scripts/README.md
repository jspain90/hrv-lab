# Automated Weather Data Updates

This directory contains scripts to automatically fetch weather data daily.

## Setup (One-time)

### Option 1: Automatic Setup (Recommended)

1. Right-click on `setup_daily_weather_task.ps1`
2. Select "Run with PowerShell"
3. If prompted, allow administrator access
4. Follow the prompts

The script will:
- Create a scheduled task that runs daily at 6:00 AM
- Configure it to fetch the last 7 days of weather data
- Set up logging to the `logs/` directory

### Option 2: Manual Setup

If you prefer to create the task manually:

1. Open Task Scheduler (search for "Task Scheduler" in Start menu)
2. Click "Create Task" (not "Create Basic Task")
3. **General Tab:**
   - Name: `HRV-Lab Daily Weather Update`
   - Description: `Fetches daily weather data from Open-Meteo API`
   - Select "Run whether user is logged on or not"

4. **Triggers Tab:**
   - Click "New..."
   - Begin the task: "On a schedule"
   - Settings: Daily, at 6:00 AM
   - Check "Enabled"

5. **Actions Tab:**
   - Click "New..."
   - Action: "Start a program"
   - Program/script: Browse to `scripts/daily_weather_update.bat`
   - Start in: Your HRV-Lab project root directory

6. **Settings Tab:**
   - Check "Allow task to be run on demand"
   - Check "Run task as soon as possible after a scheduled start is missed"
   - Check "Start the task only if the computer is on AC power" (uncheck if laptop)

## Testing

To test the scheduled task immediately:

```powershell
Start-ScheduledTask -TaskName "HRV-Lab Daily Weather Update"
```

Or right-click the task in Task Scheduler and select "Run".

Check the logs in `logs/weather_update_YYYY-MM-DD.log` to verify it worked.

## Manual Execution

You can also run the weather update manually anytime:

```batch
scripts\daily_weather_update.bat
```

Or use the Python script directly:

```bash
python pipeline/weather_fetch.py --days 7
```

## Logs

Daily execution logs are stored in `logs/weather_update_YYYY-MM-DD.log`

Each log includes:
- Start/end timestamps
- Weather fetch output
- Success/error status

## Troubleshooting

**Task doesn't run:**
- Check Task Scheduler for error messages
- Verify Python is in your PATH
- Ensure `requests` library is installed: `pip install requests`

**No data fetched:**
- Check the log file for error messages
- Verify internet connection
- Test manual execution: `python pipeline/weather_fetch.py --days 1`

**Permission errors:**
- Ensure the task runs with your user account
- Check that the project directory has write permissions

## Modifying the Schedule

To change when the task runs:

1. Open Task Scheduler
2. Find "HRV-Lab Daily Weather Update"
3. Right-click → Properties
4. Go to "Triggers" tab
5. Edit the existing trigger

Or re-run `setup_daily_weather_task.ps1` and modify the `$TriggerTime` variable before running.
