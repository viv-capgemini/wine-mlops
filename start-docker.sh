#!/bin/bash

# Script to start wine-mlops services with Docker Compose

set -e

echo "================================"
echo "Wine MLOps Docker Setup"
echo "================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please update .env with your AWS credentials and database settings"
    exit 1
fi

echo "Building images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo ""
echo "================================"
echo "Services started!"
echo "================================"
echo ""
echo "MLflow UI:  http://localhost:5000"
echo "PostgreSQL: localhost:5432"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f mlflow      # MLflow server"
echo "  docker-compose logs -f train       # Training service"
echo "  docker-compose logs -f postgres    # PostgreSQL"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To run training:"
echo "  docker-compose run train python train.py --experiment wine-prediction --run docker-run-1"
echo ""
