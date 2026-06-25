@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dashboard.ps1"
if errorlevel 1 (
    echo.
    echo The Recruiter Dashboard could not be started.
    echo Review logs\streamlit-error.log for details.
    pause
    exit /b 1
)
start "" "http://localhost:8501"
endlocal
