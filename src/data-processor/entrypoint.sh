#!/bin/bash

echo "Preprocess container is running!!!"
# Activate the Pipenv virtual environment and install dependencies
# pipenv run pip install -r requirements.txt

# pipenv run pip install --force-reinstall opencv-python-headless

# Authenticate gcloud using service account
# gcloud auth activate-service-account --key-file=secrets/ml-workflow.json

# Set GCP Project Details
gcloud config set project $GCP_PROJECT

# Run the preprocess.py script
# pipenv run python model.py
# pipenv run bash package-trainer.sh

# pipenv run python cli.py

pipenv shell
