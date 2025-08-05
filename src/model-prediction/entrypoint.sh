#!/bin/bash

echo "Inference container is running!!!"
# Activate the Pipenv virtual environment and install dependencies
pipenv run pip install -r requirements.txt

args="$@"
echo $args

if [[ -z ${args} ]]; 
then
    # Authenticate gcloud using service account
    gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
    # Set GCP Project Details
    gcloud config set project $GCP_PROJECT
    #/bin/bash
    pipenv shell
else
  pipenv run python $args
fi

# # Authenticate gcloud using service account
# # gcloud auth activate-service-account --key-file=secrets/ml-workflow.json

# # Set GCP Project Details
# gcloud config set project $GCP_PROJECT

# # Run the preprocess.py script
# # pipenv run python model.py
# # pipenv run bash package-trainer.sh

# pipenv run python cli.py --test

# #pipenv shell