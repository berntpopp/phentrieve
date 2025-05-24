# Docker & NPM Deployment

This page explains how to deploy Phentrieve using Docker for the backend and NPM for frontend development.

## Prerequisites

- Docker and Docker Compose installed
- Node.js and NPM installed (for frontend development)
- Git for cloning the repository

## Backend Deployment with Docker

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

- `PHENTRIEVE_HOST_DATA_DIR`: Host directory for storing data
- `API_PORT`: Port for the API (default: 8000)
- Other settings as needed

### Step 3: Build and Start the API Container

```bash
docker-compose up -d api
```

This will:
1. Build the API Docker image
2. Start the API container
3. Expose the API on the configured port

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

## Frontend Development with NPM

For frontend development, you can use NPM directly without Docker:

### Step 1: Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 2: Configure Frontend Environment

Create a `.env` file in the frontend directory:

```
VUE_APP_API_URL=http://localhost:8000
```

### Step 3: Start Development Server

```bash
npm run serve
```

This will start a development server with hot-reloading at http://localhost:8080.

## Building Frontend for Production

When you're ready to build the frontend for production:

```bash
cd frontend
npm run build
```

This will generate production files in the `dist` directory, which can be served with any web server.

## Combined Docker Deployment

For a full deployment using Docker for both backend and frontend:

```bash
# Build and start all containers
docker-compose up -d

# Prepare data and build indexes
docker-compose exec api bash -c "phentrieve data prepare && phentrieve index build --model-name \"FremyCompany/BioLORD-2023-M\""
```

This will:
1. Build and start the API container
2. Build and start the frontend container with Nginx
3. Configure networking between the containers

## Updating the Deployment

To update your deployment:

```bash
# Pull the latest code
git pull

# Rebuild and restart containers
docker-compose down
docker-compose up -d --build
```
