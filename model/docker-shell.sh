#!/bin/bash

set -e

export IMAGE_NAME="capy-model"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/secrets/
export GCS_BUCKET_URI="gs://capy-data"
export GCP_PROJECT="psychic-bedrock-398320"
export WANDB_KEY="6a862c7a22f68c00ceb59a5daf60d10ae341fb94"


# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
# docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS="/secrets/data-pipeline.json" \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME