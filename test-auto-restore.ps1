# Test script for auto-restore functionality
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Auto-Restore Test Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check if backups directory exists
Write-Host "1. Checking backups directory..." -ForegroundColor Yellow
if (Test-Path "./backups") {
    Write-Host "OK backups/ directory exists" -ForegroundColor Green
} else {
    Write-Host "ERROR backups/ directory not found" -ForegroundColor Red
    $allGood = $false
}

# Check for PostgreSQL backup
Write-Host ""
Write-Host "2. Checking for PostgreSQL backup..." -ForegroundColor Yellow
$pgBackups = Get-ChildItem -Path "./backups" -Filter "postgres-*.sql" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($pgBackups) {
    $pgBackup = $pgBackups[0]
    Write-Host "OK Found: $($pgBackup.Name)" -ForegroundColor Green
} else {
    Write-Host "WARNING No PostgreSQL backup found" -ForegroundColor Yellow
    $allGood = $false
}

# Check for Weaviate backup
Write-Host ""
Write-Host "3. Checking for Weaviate backup..." -ForegroundColor Yellow
$wvBackups = Get-ChildItem -Path "./backups" -Filter "weaviate-*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($wvBackups) {
    $wvBackup = $wvBackups[0]
    Write-Host "OK Found: $($wvBackup.Name)" -ForegroundColor Green
} else {
    Write-Host "WARNING No Weaviate backup found" -ForegroundColor Yellow
    $allGood = $false
}

# Check docker-compose.coolify.yml volume mount
Write-Host ""
Write-Host "4. Checking docker-compose.coolify.yml volume mount..." -ForegroundColor Yellow
$dockerComposeCoolify = Get-Content "docker-compose.coolify.yml" -Raw
if ($dockerComposeCoolify -match "backups:/app/backups") {
    Write-Host "OK Volume mount found" -ForegroundColor Green
} else {
    Write-Host "ERROR Volume mount missing" -ForegroundColor Red
    $allGood = $false
}

# Summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if ($allGood -and $pgBackups -and $wvBackups) {
    Write-Host ""
    Write-Host "OK Auto-restore is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. git add backups/"
    Write-Host "2. git commit -m 'Add database backups for auto-restore'"
    Write-Host "3. git push origin main"
    Write-Host "4. Deploy to Coolify"
} else {
    Write-Host ""
    Write-Host "WARNING Auto-restore is not ready" -ForegroundColor Yellow
    Write-Host "Fix the issues above and run this script again."
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
