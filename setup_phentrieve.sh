#!/bin/bash

# setup_phentrieve.sh - Comprehensive setup script for Phentrieve with Docker and NPM
# This script automates initial setup for Phentrieve deployment with Docker and Nginx Proxy Manager

# Exit on error
set -e

# Print colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Phentrieve Docker + NPM Setup ===${NC}"
echo "This script will prepare your environment for running Phentrieve with Nginx Proxy Manager."

# --- Dependency Checks ---
echo -e "\n${YELLOW}Step 0: Checking Dependencies...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed.${NC}"

# Determine docker compose command (V1 vs V2)
if docker compose version &> /dev/null; then
    COMPOSE_COMMAND="docker compose"
    echo "Using Docker Compose V2 (docker compose plugin)."
elif command -v docker-compose &> /dev/null; then
    COMPOSE_COMMAND="docker-compose"
    echo "Using Docker Compose V1 (docker-compose command)."
else
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is available (using '$COMPOSE_COMMAND').${NC}"

# Ensure we're in the repository root directory
if [ ! -f "docker-compose.yml" ] || [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Key project files (docker-compose.yml, pyproject.toml) not found.${NC}"
    echo "Please ensure you are running this script from the Phentrieve repository root directory."
    exit 1
fi
echo -e "${GREEN}✓ Running from project root directory.${NC}"


# --- Environment Configuration ---
echo -e "\n${YELLOW}Step 1: Checking Docker Environment Configuration (.env.docker)...${NC}"
if [ ! -f ".env.docker" ]; then
    echo "'.env.docker' file not found. Creating from example..."
    if [ ! -f ".env.docker.example" ]; then
        echo -e "${RED}Error: '.env.docker.example' template not found! Cannot create .env.docker.${NC}"
        exit 1
    fi
    cp .env.docker.example .env.docker
    echo -e "${GREEN}✓ Created .env.docker from template.${NC}"
    echo -e "${YELLOW}IMPORTANT: Review and edit '.env.docker'. Ensure PHENTRIEVE_HOST_DATA_DIR is an absolute path.${NC}"
    echo "   Ensure NPM_SHARED_NETWORK_NAME is set to NPM's actual default network (e.g., 'npm_default')."
    
    if command -v nano &> /dev/null; then
        echo "Opening .env.docker in nano for you. Save (Ctrl+O, Enter) and Exit (Ctrl+X) when done."
        sleep 2
        nano .env.docker
    else
        echo "Please manually edit the '.env.docker' file now."
        read -p "Press Enter once you have edited and saved .env.docker..."
    fi
fi

echo "Loading environment variables from .env.docker..."
if [ -f ".env.docker" ]; then
    set -o allexport 
    source ".env.docker"
    set +o allexport
else
    echo -e "${RED}Error: .env.docker file is still missing after attempting to create it!${NC}"
    exit 1
fi

if [ -z "$PHENTRIEVE_HOST_DATA_DIR" ]; then
    echo -e "${RED}Error: PHENTRIEVE_HOST_DATA_DIR is not set in .env.docker.${NC}"
    exit 1
fi
if [[ "$PHENTRIEVE_HOST_DATA_DIR" != /* ]]; then
    echo -e "${RED}Error: PHENTRIEVE_HOST_DATA_DIR ('$PHENTRIEVE_HOST_DATA_DIR') in .env.docker must be an absolute path.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ PHENTRIEVE_HOST_DATA_DIR is set to: $PHENTRIEVE_HOST_DATA_DIR${NC}"

# --- Host Directory Structure ---
echo -e "\n${YELLOW}Step 2: Ensuring Host Data Directory Structure...${NC}"
HPO_CORE_DATA_SUBDIR_HOST="hpo_core_data" 
INDEXES_SUBDIR_HOST="indexes"
# TEST_CASES_SUBDIR_HOST="test_cases" # Assuming test cases are part of the repo, not dynamic data
TRANSLATIONS_SUBDIR_HOST="hpo_translations"
RESULTS_SUBDIR_HOST="results"

mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$HPO_CORE_DATA_SUBDIR_HOST"
mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$INDEXES_SUBDIR_HOST"
# mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$TEST_CASES_SUBDIR_HOST" # Only if test_cases are outside repo
mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$TRANSLATIONS_SUBDIR_HOST/de" 
mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$TRANSLATIONS_SUBDIR_HOST/en" 
mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$RESULTS_SUBDIR_HOST/summaries"
mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$RESULTS_SUBDIR_HOST/detailed"
mkdir -p "$PHENTRIEVE_HOST_DATA_DIR/$RESULTS_SUBDIR_HOST/visualizations"
echo -e "${GREEN}✓ Host data directory structure checked/created under $PHENTRIEVE_HOST_DATA_DIR.${NC}"

# --- Shared Docker Network with NPM ---
# This variable should be set in .env.docker (e.g., NPM_SHARED_NETWORK_NAME=npm_default)
if [ -z "$NPM_SHARED_NETWORK_NAME" ]; then
    echo -e "${RED}Error: NPM_SHARED_NETWORK_NAME is not set in .env.docker. Please define it (e.g., 'npm_default').${NC}"
    exit 1
fi
echo -e "\n${YELLOW}Step 3: Checking Shared Docker Network '$NPM_SHARED_NETWORK_NAME'...${NC}"
if ! docker network inspect "$NPM_SHARED_NETWORK_NAME" &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker network '$NPM_SHARED_NETWORK_NAME' does not exist.${NC}"
    echo "This script assumes NPM (Nginx Proxy Manager) has already created this network."
    echo "If NPM is running and uses a different network name, update NPM_SHARED_NETWORK_NAME in .env.docker."
    echo "If NPM is not yet running or uses a different setup, you might need to create this network manually"
    echo "or ensure NPM's docker-compose.yml will create/use it, AND that Phentrieve's docker-compose.yml"
    echo "refers to it correctly as an external network."
    # For Option 1 (using NPM's default), we assume it exists. If not, user intervention is needed.
else
    echo -e "${GREEN}✓ Docker network '$NPM_SHARED_NETWORK_NAME' found.${NC}"
fi

# --- Build Phentrieve API Image ---
echo -e "\n${YELLOW}Step 4: Building Phentrieve API Docker image (if needed)...${NC}"
$COMPOSE_COMMAND -f docker-compose.yml --env-file .env.docker build phentrieve_api
echo -e "${GREEN}✓ Phentrieve API image build process completed.${NC}"

# --- HPO Data Preparation ---
CONTAINER_DATA_ROOT_MOUNT_POINT="/phentrieve_data_mount" 
CONTAINER_HPO_CORE_DATA_DIR="$CONTAINER_DATA_ROOT_MOUNT_POINT/$HPO_CORE_DATA_SUBDIR_HOST"

echo -e "\n${YELLOW}Step 5: Checking and Preparing HPO Core Data...${NC}"
HPO_JSON_FILE_HOST="$PHENTRIEVE_HOST_DATA_DIR/$HPO_CORE_DATA_SUBDIR_HOST/hp.json" 

if [ ! -f "$HPO_JSON_FILE_HOST" ]; then 
    echo "HPO core data file (hp.json) not found on host at $HPO_JSON_FILE_HOST."
    echo "Running 'phentrieve data prepare' inside a Docker container..."
    
    $COMPOSE_COMMAND -f docker-compose.yml --env-file .env.docker run --rm \
        phentrieve_api phentrieve data prepare --force \
        --data-dir "$CONTAINER_HPO_CORE_DATA_DIR" 
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ HPO core data preparation completed successfully.${NC}"
    else
        echo -e "${RED}ERROR: HPO data preparation failed. Check container logs for details.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ HPO core data files found on host. Skipping preparation.${NC}"
fi

# --- Default Model Indexing ---
echo -e "\n${YELLOW}Step 6: Checking and Building Default Model Index...${NC}"
DEFAULT_MODEL_FOR_INDEXING=${DEFAULT_SETUP_INDEX_MODEL:-"FremyCompany/BioLORD-2023-M"}
echo "Default model for initial index check: $DEFAULT_MODEL_FOR_INDEXING"

echo "Determining model slug for $DEFAULT_MODEL_FOR_INDEXING..."
MODEL_SLUG_RAW=$($COMPOSE_COMMAND -f docker-compose.yml --env-file .env.docker run --rm \
    phentrieve_api python -c "from phentrieve.utils import get_model_slug; print(get_model_slug('$DEFAULT_MODEL_FOR_INDEXING'))")
MODEL_SLUG=$(echo "$MODEL_SLUG_RAW" | tr -d '\r\n')

if [ -z "$MODEL_SLUG" ]; then
    echo -e "${RED}ERROR: Could not determine model slug for '$DEFAULT_MODEL_FOR_INDEXING'. Output was: '$MODEL_SLUG_RAW'${NC}"
    exit 1
fi
echo "Model slug determined as: '$MODEL_SLUG'"

INDEX_DIR_HOST="$PHENTRIEVE_HOST_DATA_DIR/$INDEXES_SUBDIR_HOST/phentrieve_$MODEL_SLUG"

if [ ! -d "$INDEX_DIR_HOST" ] || [ -z "$(ls -A "$INDEX_DIR_HOST" 2>/dev/null)" ]; then
    echo "Index for default model '$DEFAULT_MODEL_FOR_INDEXING' not found or empty in $INDEX_DIR_HOST."
    echo "Running 'phentrieve index build' for the default model inside a Docker container..."
    
    $COMPOSE_COMMAND -f docker-compose.yml --env-file .env.docker run --rm \
        phentrieve_api phentrieve index build --model-name "$DEFAULT_MODEL_FOR_INDEXING" --recreate
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Default model index building completed successfully.${NC}"
    else
        echo -e "${RED}ERROR: Default model index building failed. Check container logs for details.${NC}"
    fi
else
    echo -e "${GREEN}✓ Default model index for '$DEFAULT_MODEL_FOR_INDEXING' found on host at $INDEX_DIR_HOST. Skipping index build.${NC}"
fi

# --- Final Instructions ---
echo ""
echo -e "${GREEN}=== Phentrieve Setup Script Finished ===${NC}"
echo -e "\n${YELLOW}IMPORTANT NEXT STEPS:${NC}"
echo "1.  **DNS Configuration:** Ensure your DNS records point to this server's public IP for the domains:"
VITE_FRONTEND_URL_PUBLIC_VAL=${VITE_FRONTEND_URL_PUBLIC:-"YOUR_FRONTEND_DOMAIN (e.g., phentrieve.example.com)"}
VITE_API_URL_PUBLIC_VAL=${VITE_API_URL_PUBLIC:-"YOUR_API_DOMAIN (e.g., api.phentrieve.example.com)"} # Show base domain for API
VITE_API_URL_PUBLIC_BASE=$(echo $VITE_API_URL_PUBLIC_VAL | sed -E 's|/api/v1/?$||')

echo "    - Frontend: $VITE_FRONTEND_URL_PUBLIC_VAL"
echo "    - API:      $VITE_API_URL_PUBLIC_BASE"


echo -e "\n2.  **Nginx Proxy Manager Configuration (Web UI):**"
echo "    a. Access your NPM Web UI."
echo "    b. Ensure NPM's 'app' service is connected to the '$NPM_SHARED_NETWORK_NAME' Docker network."
echo "       (You might need to edit NPM's docker-compose.yml and restart it if not already done)."
echo "    c. Add a Proxy Host for the Phentrieve Frontend:"
echo "       - Domain Names: $(echo $VITE_FRONTEND_URL_PUBLIC_VAL | sed -E 's|https?://||')"
echo "       - Scheme: http"
echo "       - Forward Hostname / IP: phentrieve_frontend (Docker service name from Phentrieve's docker-compose.yml)"
echo "       - Forward Port: 80 (Internal port of frontend's Nginx container)"
echo "       - SSL: Request a new SSL Certificate, enable 'Force SSL'."
echo "    d. Add another Proxy Host for the Phentrieve API:"
echo "       - Domain Names: $(echo $VITE_API_URL_PUBLIC_BASE | sed -E 's|https?://||')"
echo "       - Scheme: http"
echo "       - Forward Hostname / IP: phentrieve_api (Docker service name)"
echo "       - Forward Port: 8000 (Internal port of API's Uvicorn container)"
echo "       - SSL: Request a new SSL Certificate, enable 'Force SSL'."

echo -e "\n3.  **Start Phentrieve Application Stack:**"
echo "    Once DNS is propagated and NPM is configured, run from Phentrieve project root:"
echo "    $COMPOSE_COMMAND -f docker-compose.yml --env-file .env.docker up -d --build"

echo -e "\nAccess Phentrieve via your configured frontend domain (e.g., $VITE_FRONTEND_URL_PUBLIC_VAL)."
echo "API Swagger UI via its domain (e.g., $(echo $VITE_API_URL_PUBLIC_BASE)/docs)."