#!/bin/bash
# Fast local development environment setup
# This script sets up everything needed for native local development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Phentrieve Fast Local Development Environment Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status messages
print_status() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python
if ! command_exists python3 && ! command_exists python; then
    print_error "Python not installed. Please install Python 3.9+"
    exit 1
fi
print_success "Python found: $(python3 --version 2>&1 || python --version)"

# Check uv
if ! command_exists uv; then
    print_error "uv not installed."
    echo ""
    echo "Install uv with:"
    echo "  pip install uv"
    echo "  OR"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "uv found: $(uv --version)"

# Check Node.js
if ! command_exists node; then
    print_error "Node.js not installed. Please install Node.js 18+"
    exit 1
fi
print_success "Node.js found: $(node --version)"

# Check npm
if ! command_exists npm; then
    print_error "npm not installed. Please install npm"
    exit 1
fi
print_success "npm found: $(npm --version)"

echo ""
print_status "Installing Python dependencies with uv (fast)..."
uv sync
print_success "Python dependencies installed"

echo ""
print_status "Installing frontend dependencies..."
cd frontend
npm install
cd ..
print_success "Frontend dependencies installed"

echo ""
print_status "Verifying data directory structure..."

if [ ! -d "data" ]; then
    mkdir -p data
    print_warning "Created data/ directory"
fi

if [ ! -d "data/hpo_core_data" ]; then
    print_warning "HPO data not found."
    echo ""
    echo "You need to prepare HPO data before development:"
    echo "  phentrieve data prepare"
    echo ""
else
    print_success "HPO data found"
fi

if [ ! -d "data/indexes" ]; then
    mkdir -p data/indexes
    print_warning "Created data/indexes/ directory"
    echo ""
    echo "You may want to build vector indexes:"
    echo "  phentrieve index build"
    echo ""
else
    print_success "Indexes directory found"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Start development:"
echo ""
echo -e "  ${BLUE}Terminal 1 (API):${NC}"
echo "    make dev-api"
echo "    → http://localhost:8000"
echo "    → http://localhost:8000/docs (OpenAPI)"
echo ""
echo -e "  ${BLUE}Terminal 2 (Frontend):${NC}"
echo "    make dev-frontend"
echo "    → http://localhost:5173"
echo ""
echo "Performance:"
echo "  • API hot reload: <1s on .py changes"
echo "  • Frontend HMR: <50ms on .vue/.ts changes"
echo "  • 100x faster than Docker!"
echo ""
echo -e "${YELLOW}First time?${NC} Run these commands if needed:"
echo "  phentrieve data prepare    # Download HPO data"
echo "  phentrieve index build     # Build vector indexes"
echo ""
