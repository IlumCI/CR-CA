# Fix Docker Credential Helper for Rancher Desktop
# This script removes the problematic credential helper from Docker config

$dockerConfigPath = "$env:USERPROFILE\.docker\config.json"
$separator = "============================================================"

Write-Host "`nFixing Docker credential helper for Rancher Desktop..." -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan

# Check if Docker is available
Write-Host "`nChecking Docker availability..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Docker found: $dockerVersion" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Docker command returned error" -ForegroundColor Yellow
    }
} catch {
    Write-Host "ERROR: Docker not found in PATH. Please ensure Rancher Desktop is installed." -ForegroundColor Red
    exit 1
}

# Check if Docker daemon is running
Write-Host "`nChecking Docker daemon..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        if ($dockerInfo -match "rancher|moby") {
            Write-Host "SUCCESS: Rancher Desktop detected and running" -ForegroundColor Green
        } else {
            Write-Host "SUCCESS: Docker daemon is running" -ForegroundColor Green
        }
    } else {
        Write-Host "ERROR: Docker daemon is not running. Please start Rancher Desktop." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "ERROR: Cannot connect to Docker daemon. Please start Rancher Desktop." -ForegroundColor Red
    exit 1
}

# Fix credential helper
Write-Host "`nChecking Docker config..." -ForegroundColor Yellow

if (Test-Path $dockerConfigPath) {
    Write-Host "Found Docker config at: $dockerConfigPath" -ForegroundColor Yellow
    
    try {
        $configContent = Get-Content $dockerConfigPath -Raw
        $config = $configContent | ConvertFrom-Json
        
        if ($config.PSObject.Properties.Name -contains "credsStore") {
            $credsStoreValue = $config.credsStore
            Write-Host "WARNING: Found credential helper: $credsStoreValue" -ForegroundColor Yellow
            Write-Host "Removing credential helper..." -ForegroundColor Yellow
            
            # Backup original
            $backupPath = "$dockerConfigPath.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
            Copy-Item $dockerConfigPath $backupPath -Force
            Write-Host "SUCCESS: Backup saved to: $backupPath" -ForegroundColor Green
            
            # Remove credsStore
            $config.PSObject.Properties.Remove('credsStore')
            
            # Save updated config with proper formatting
            # Use -Compress to avoid formatting issues, then format manually for readability
            $updatedConfig = $config | ConvertTo-Json -Depth 10 -Compress
            $formattedConfig = $updatedConfig | ConvertFrom-Json | ConvertTo-Json -Depth 10
            
            # Write with UTF8NoBOM to avoid BOM issues that can break JSON parsing
            $utf8NoBom = New-Object System.Text.UTF8Encoding $false
            [System.IO.File]::WriteAllText($dockerConfigPath, $formattedConfig, $utf8NoBom)
            
            # Verify the written file is valid JSON
            try {
                $verifyConfig = Get-Content $dockerConfigPath -Raw | ConvertFrom-Json
                Write-Host "SUCCESS: Credential helper removed successfully!" -ForegroundColor Green
            } catch {
                Write-Host "ERROR: Written config is invalid JSON. Restoring backup..." -ForegroundColor Red
                Copy-Item $backupPath $dockerConfigPath -Force
                Write-Host "ERROR: Failed to fix config. Please edit manually." -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "SUCCESS: No credential helper found - config looks good!" -ForegroundColor Green
        }
    } catch {
        Write-Host "ERROR: Error processing config: $_" -ForegroundColor Red
        Write-Host "TIP: You can manually edit: $dockerConfigPath" -ForegroundColor Yellow
        Write-Host "    Remove the line: `"credsStore`": `"wincred`"" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "Docker config not found at: $dockerConfigPath" -ForegroundColor Yellow
    Write-Host "Creating new config without credential helper..." -ForegroundColor Cyan
    
    $dockerDir = Split-Path $dockerConfigPath
    if (-not (Test-Path $dockerDir)) {
        New-Item -ItemType Directory -Path $dockerDir -Force | Out-Null
    }
    
    $newConfig = @{
        auths = @{}
    } | ConvertTo-Json -Depth 10
    
    # Write with UTF8NoBOM to avoid BOM issues
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($dockerConfigPath, $newConfig, $utf8NoBom)
    
    # Verify the written file is valid JSON
    try {
        $verifyConfig = Get-Content $dockerConfigPath -Raw | ConvertFrom-Json
        Write-Host "SUCCESS: Created new Docker config without credential helper!" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Created config is invalid JSON." -ForegroundColor Red
        exit 1
    }
}

# Verify the fix
Write-Host "`nVerifying fix..." -ForegroundColor Yellow
try {
    $testConfig = Get-Content $dockerConfigPath -Raw | ConvertFrom-Json
    if ($testConfig.PSObject.Properties.Name -contains "credsStore") {
        Write-Host "ERROR: Credential helper still present!" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "SUCCESS: Config verified - no credential helper found" -ForegroundColor Green
        Write-Host "SUCCESS: JSON is valid and properly formatted" -ForegroundColor Green
    }
} catch {
    Write-Host "ERROR: Config file is invalid JSON: $_" -ForegroundColor Red
    Write-Host "Attempting to restore from backup..." -ForegroundColor Yellow
    $backupFiles = Get-ChildItem "$dockerConfigPath.backup.*" | Sort-Object LastWriteTime -Descending
    if ($backupFiles) {
        $latestBackup = $backupFiles[0].FullName
        Copy-Item $latestBackup $dockerConfigPath -Force
        Write-Host "SUCCESS: Restored from backup: $latestBackup" -ForegroundColor Green
        Write-Host "Please restart Rancher Desktop and try again." -ForegroundColor Yellow
    } else {
        Write-Host "ERROR: No backup found. Please manually fix the config file." -ForegroundColor Red
    }
    exit 1
}

Write-Host "`n$separator" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "   1. Restart Rancher Desktop (if it was running)" -ForegroundColor White
Write-Host "   2. Verify: docker info" -ForegroundColor White
Write-Host "   3. Test: docker compose version" -ForegroundColor White
Write-Host "   4. Run the daemon: python corposwarm.py --daemon --verbose" -ForegroundColor White
$message = "`nSUCCESS: Fix complete! You can now use bolt.diy with Rancher Desktop."
Write-Host $message -ForegroundColor Green

