param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$PythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "Khong tim thay .venv\Scripts\python.exe. Hay tao moi truong ao truoc!" -ForegroundColor Red
    exit 1
}

$Url = "http://127.0.0.1:$Port"
Write-Host "Khoi dong E-AM for VRPRB Web Dashboard tai $Url (nhan Ctrl+C de dung)..." -ForegroundColor Cyan

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($TargetUrl)
        Start-Sleep -Seconds 2
        Start-Process $TargetUrl
    } -ArgumentList $Url | Out-Null
}

& $PythonExe -m web.server --host 127.0.0.1 --port $Port
