#!/bin/bash
# Script to attempt Docker Hub login if needed
# This can help with image pull rate limits

# Check if already logged in
if ! docker info 2>/dev/null | grep -q "Username"; then
  echo "You're not logged into Docker Hub. Pull rate limits may apply."
  echo "If you have a Docker Hub account, you can log in to increase rate limits."
  echo "Would you like to log in to Docker Hub? (y/n)"
  read -r response
  
  if [[ "$response" =~ ^[Yy]$ ]]; then
    docker login
  else
    echo "Continuing without Docker Hub login. Pull operations may be limited."
  fi
else
  echo "Already logged into Docker Hub."
fi
