$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDirectory = Join-Path $projectRoot "logs"
$stdoutLog = Join-Path $logDirectory "streamlit.log"
$stderrLog = Join-Path $logDirectory "streamlit-error.log"
$port = 8501

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

try {
    $existingHealth = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/_stcore/health" -TimeoutSec 2
    if ($existingHealth.StatusCode -eq 200) {
        Write-Output "RedRob Recruiter Dashboard is already running at http://localhost:$port"
        exit 0
    }
}
catch {
    # No healthy dashboard is currently running.
}

$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @(
    "-m", "streamlit", "run",
    (Join-Path $projectRoot "app\streamlit_app.py"),
    "--server.headless", "true",
    "--server.address", "127.0.0.1",
    "--server.port", "$port",
    "--browser.gatherUsageStats", "false"
)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        $errorOutput = if (Test-Path $stderrLog) { Get-Content -Raw $stderrLog } else { "No error log was produced." }
        throw "Streamlit exited during startup.`n$errorOutput"
    }

    try {
        $health = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/_stcore/health" -TimeoutSec 2
        if ($health.StatusCode -eq 200) {
            Write-Output "RedRob Recruiter Dashboard is running at http://localhost:$port"
            Write-Output "Process ID: $($process.Id)"
            Write-Output "Logs: $stdoutLog"
            exit 0
        }
    }
    catch {
        # The application may still be starting.
    }
}

throw "Streamlit did not become healthy. Review $stderrLog"
