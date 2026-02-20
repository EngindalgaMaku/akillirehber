#!/bin/bash
# Database reset script for Coolify deployment
# Run this on the server to reset the database

echo "Stopping containers..."
docker-compose -f docker-compose.coolify.yml down

echo "Removing postgres volume..."
docker volume rm $(docker volume ls -q | grep postgres) 2>/dev/null || echo "No postgres volume found"

echo "Starting containers..."
docker-compose -f docker-compose.coolify.yml up -d

echo "Database reset complete!"
