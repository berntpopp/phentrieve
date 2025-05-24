# Deployment

This section covers how to deploy Phentrieve in various environments, from local development to production systems.

## Deployment Options

Phentrieve can be deployed in several ways, depending on your needs:

1. **Local Installation**: Install the Python package locally for development or personal use
2. **Docker Containers**: Deploy as containerized services for easier management and isolation
3. **Production Deployment**: Set up a full production environment with proper networking and security

## Components

A complete Phentrieve deployment consists of:

- **Core Python Package**: The foundation that provides all functionality
- **API Server**: A FastAPI-based service that exposes endpoints for querying and processing
- **Frontend**: A Vue.js-based web interface for user-friendly interaction
- **Vector Database**: ChromaDB for storing and retrieving vector embeddings
- **Data Storage**: Directories for HPO data, indexes, and results

## Section Contents

- [Docker & NPM](docker-npm.md): Setting up Phentrieve using Docker and NPM for frontend development
- [Local Docker](local-docker.md): Running Phentrieve locally with Docker Compose
- [Data Management](data-management.md): Managing Phentrieve data in different deployment scenarios
