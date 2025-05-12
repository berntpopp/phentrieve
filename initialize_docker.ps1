# Initialize Docker environment for Phentrieve app
# This script launches Docker containers using existing data folders

# Check that required data directories exist
if (-Not (Test-Path -Path "data")) {
    Write-Host "ERROR: The data directory does not exist. Please create it first." -ForegroundColor Red
    exit 1
}

# Ensure the indexes directory exists inside the data directory
if (-Not (Test-Path -Path "data/indexes")) {
    Write-Host "WARNING: The indexes directory (data/indexes) does not exist." -ForegroundColor Yellow
    Write-Host "You may need to run the indexing script first." -ForegroundColor Yellow
}

# Pull Docker images first to avoid credential issues
Write-Host "Pulling required Docker images..."
docker pull python:3.9-slim-bullseye
if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: Failed to pull Python image, will attempt to build anyway." -ForegroundColor Yellow }

docker pull node:lts-alpine
if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: Failed to pull Node image, will attempt to build anyway." -ForegroundColor Yellow }

docker pull nginx:stable-alpine
if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: Failed to pull Nginx image, will attempt to build anyway." -ForegroundColor Yellow }

# Build and start the Docker containers
Write-Host "Building and starting Docker containers..."
docker-compose up --build -d

Write-Host "Docker containers are now running:"
Write-Host "  - API: http://localhost:8001"
Write-Host "  - Frontend: http://localhost:8080"
Write-Host ""
Write-Host "To view logs: docker-compose logs -f"
Write-Host "To stop containers: docker-compose down"
