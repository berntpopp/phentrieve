#!/bin/bash
# Initialize Docker environment for Phentrieve app on Linux
# This script launches Docker containers using existing data folders

# Check that required data directories exist
if [ ! -d "data" ]; then
    echo "ERROR: The data directory does not exist. Please create it first."
    exit 1
fi

# Ensure the indexes directory exists inside the data directory
if [ ! -d "data/indexes" ]; then
    echo "WARNING: The indexes directory (data/indexes) does not exist."
    echo "You may need to run the indexing script first."
fi

# Make sure the data directory has the right permissions
# Docker runs as root inside the container, but we need wider permissions
chmod -R 777 data

# Pull Docker images first to avoid credential issues
echo "Pulling required Docker images..."
docker pull python:3.9-slim-bullseye || { echo "WARNING: Failed to pull Python image, will attempt to build anyway."; }
docker pull node:lts-alpine || { echo "WARNING: Failed to pull Node image, will attempt to build anyway."; }
docker pull nginx:stable-alpine || { echo "WARNING: Failed to pull Nginx image, will attempt to build anyway."; }

# Build and start the Docker containers
echo "Building and starting Docker containers..."
docker-compose up --build -d

echo "Docker containers are now running:"
echo "  - API: http://localhost:8001"
echo "  - Frontend: http://localhost:8080"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop containers: docker-compose down"
