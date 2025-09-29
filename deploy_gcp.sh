#!/bin/bash

# Sapheneia TimesFM GCP Deployment Script

set -e

PROJECT_ID="${1:-your-project-id}"
REGION="${2:-us-central1}"

if [[ "$PROJECT_ID" == "your-project-id" ]]; then
    echo "Usage: ./deploy_gcp.sh YOUR_PROJECT_ID [REGION]"
    echo "Example: ./deploy_gcp.sh sapheneia-demo us-central1"
    exit 1
fi

echo "🚀 Deploying Sapheneia TimesFM to GCP"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo "Starting Cloud Build deployment..."
gcloud builds submit webapp/ --config=webapp/cloudbuild.yaml \
    --substitutions=_REGION=$REGION

# Get the service URL
SERVICE_URL=$(gcloud run services describe sapheneia-timesfm --region=$REGION --format="value(status.url)")

echo "✅ Deployment completed!"
echo "🌐 Service URL: $SERVICE_URL"
