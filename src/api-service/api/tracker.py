import os
import traceback
import asyncio
from glob import glob
import json
import pandas as pd

import tensorflow as tf
from google.cloud import storage


# bucket_name = os.environ["GCS_BUCKET_NAME"]
bucket_name = 'capy-data'
local_model_path = "/local_model"
secret_path = '/secrets/capy-key.json'

with open('/secrets/capy-key.json') as json_file:
    key_info = json.load(json_file)

# print(key_info)

# Setup experiments folder
if not os.path.exists(local_model_path):
    os.mkdir(local_model_path)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client.from_service_account_info(key_info)

    # crediental 
    # storage_client = storage.Client.from_service_account_json("C:/Users/chuqi/ac215/kaggle-data/aslfr-isolated/psychic-bedrock-398320-e41cc1b33701.json")

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def download_best_model():
    print("Download best model")
    try:
        download_file = 'asl_model2.h5'
        download_blob(
            bucket_name,
            download_file,
            os.path.join(local_model_path, download_file),
        )

    except:
        print("Error in download_best_model")
        traceback.print_exc()


class TrackerService:
    def __init__(self):
        self.timestamp = 0

    async def track(self):
        # while True:
        await asyncio.sleep(10)
        print("Download Model...")

        download_best_model()