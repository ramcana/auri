# PowerShell script to start all voice bot services

Write-Host "Starting Voice Bot Services..." -ForegroundColor Cyan

# Create a function to start a service in a new PowerShell window
function Start-ServiceWindow {
    param (
        [string]$Title,
        [string]$Command,
        [string]$WorkingDirectory
    )
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting $Title...' -ForegroundColor Green; Set-Location '$WorkingDirectory'; $Command"
}

# Start the TTS service
Start-ServiceWindow -Title "TTS Service" -Command "python ./tts_service/app.py" -WorkingDirectory "."

# Wait a moment to ensure TTS service starts first
Start-Sleep -Seconds 2

# Start the LLM service
Start-ServiceWindow -Title "LLM Service" -Command "python ./backend/llm_service.py" -WorkingDirectory "."

# Wait a moment to ensure LLM service starts
Start-Sleep -Seconds 2

# Start the Voice Chat Connector
Start-ServiceWindow -Title "Voice Chat Connector" -Command "python ./backend/voice_chat_connector.py" -WorkingDirectory "."

# Wait a moment to ensure connector starts
Start-Sleep -Seconds 2

# Start the frontend
Start-ServiceWindow -Title "Frontend" -Command "npm start" -WorkingDirectory "."

Write-Host "All services started!" -ForegroundColor Green
