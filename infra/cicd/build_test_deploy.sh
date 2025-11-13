#!/bin/bash

Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting build, test, and deploy process..."

Define variables
IMAGE_NAME="genai-assistant"
IMAGE_TAG="latest"
DOCKERFILE="./infra/docker/Dockerfile.assistant"
REQUIREMENTS_FILE="./GenAI_Factory/requirements.txt"
TEST_DIR="./GenAI_Factory/domain_models/genai_assistant/tests/"

--- Build Docker Image ---
echo "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
docker build -t $IMAGE_NAME:$IMAGE_TAG -f $DOCKERFILE .

echo "Docker image built successfully."

--- Run Tests ---
echo "Running Python tests..."

Create a temporary container to run tests in a consistent environment
CONTAINER_ID=$(docker create $IMAGE_NAME:$IMAGE_TAG)
docker cp $REQUIREMENTS_FILE $CONTAINER_ID:/app/
docker cp $TEST_DIR $CONTAINER_ID:/app/
docker start $CONTAINER_ID
docker exec $CONTAINER_ID pip install -r /app/requirements.txt
docker exec $CONTAINER_ID pytest /app/tests/

echo "Tests passed."

--- Deploy to Kubernetes (requires kubectl context) ---
echo "Deploying to Kubernetes..."
kubectl apply -f ./infra/k8s/deployments/
kubectl apply -f ./infra/k8s/services/
kubectl apply -f ./infra/k8s/configmaps/
kubectl apply -f ./infra/k8s/secrets/ # Be careful with this, secrets should be managed more securely
kubectl apply -f ./infra/k8s/ingress/

echo "Deployment finished."

echo "CI/CD process completed successfully."