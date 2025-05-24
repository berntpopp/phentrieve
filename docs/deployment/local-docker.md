# Local Docker Deployment

This page explains how to deploy Phentrieve locally using Docker Compose, which is ideal for testing and personal use.

## Prerequisites

- Docker and Docker Compose installed
- Git for cloning the repository
- Approximately 2GB free disk space for Docker images
- Additional space for HPO data and indexes (varies by model)

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/berntpopp/phentrieve.git
cd phentrieve
```

### Step 2: Configure Environment

Copy the example environment file:

```bash
cp .env.docker.example .env.docker
```

Edit the `.env.docker` file to configure:

```
# Host directory for Phentrieve data (adjust to your preferred location)
PHENTRIEVE_HOST_DATA_DIR=/path/to/your/data

# API port (default: 8000)
API_PORT=8000

# Frontend port (default: 8080)
FRONTEND_PORT=8080
```

### Step 3: Start the Containers

```bash
docker-compose up -d
```

This command starts both the API and frontend containers in detached mode.

### Step 4: Prepare Data and Build Indexes

```bash
# Enter the API container
docker-compose exec api bash

# Inside the container, prepare data and build indexes
phentrieve data prepare
phentrieve index build --model-name "FremyCompany/BioLORD-2023-M"

# Exit the container
exit
```

### Step 5: Access the Application

- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Frontend: http://localhost:8080

## Development Mode

For local development with Docker, use the development override file:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

This configuration:
- Mounts the source code directories into the containers
- Enables hot-reloading for the frontend
- Provides more verbose logging

## Common Operations

### Viewing Logs

```bash
# View logs from all containers
docker-compose logs

# View logs from a specific container
docker-compose logs api
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f
```

### Stopping the Containers

```bash
docker-compose down
```

### Updating Phentrieve

```bash
# Pull the latest code
git pull

# Rebuild and restart containers
docker-compose down
docker-compose up -d --build
```

## Troubleshooting

### Container Startup Issues

If containers fail to start:

```bash
# Check container status
docker-compose ps

# View detailed startup logs
docker-compose logs
```

### Data Directory Permissions

If you encounter permission issues with the data directory:

```bash
# Set appropriate permissions on host data directory
sudo chown -R 1000:1000 /path/to/your/data
```

### Network Issues

If containers can't communicate:

```bash
# Check if containers are on the same network
docker network ls
docker network inspect phentrieve_default
```

## Performance Tuning

For better performance with Docker:

1. Allocate sufficient resources to Docker (in Docker Desktop settings)
2. Use volume mounts for data directories to reduce I/O overhead
3. Consider using a dedicated SSD for Docker data
4. Enable GPU support if available (see Docker GPU documentation)
