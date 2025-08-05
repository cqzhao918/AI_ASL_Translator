# List of prebuilt containers for training
# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers

export GCS_BUCKET_URI="gs://capy-data/model"
export UUID=$(openssl rand -hex 6)
export DISPLAY_NAME="capy_training_job_$UUID"
export MACHINE_TYPE="n1-standard-4"
export REPLICA_COUNT=1
# export EXECUTOR_IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest"
export PYTHON_PACKAGE_URI=$GCS_BUCKET_URI/capy-trainer.tar.gz
export PYTHON_MODULE="capy-trainer.task"
# export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
# export ACCELERATOR_COUNT=1
export GCP_REGION="us-central1" # Adjust region based on you approved quotas for GPUs
export WANDB_KEY="6a862c7a22f68c00ceb59a5daf60d10ae341fb94"

export CMDARGS="--wandb_key=$WANDB_KEY"
#export CMDARGS="--model_name=mobilenetv2,--train_base,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY"
#export CMDARGS="--model_name=tfhub_mobilenetv2,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY"
#export CMDARGS="--model_name=tfhub_mobilenetv2,--train_base,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY"

# gcloud ai custom-jobs create \
#   --region=$GCP_REGION \
#   --display-name=$DISPLAY_NAME \
#   --python-package-uris=$PYTHON_PACKAGE_URI \
#   --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERATOR_COUNT,executor-image-uri=$EXECUTOR_IMAGE_URI,python-module=$PYTHON_MODULE \
#   --args=$CMDARGS


# Run training with No GPU
export EXECUTOR_IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest"
# gcloud ai custom-jobs create \
#   --region=$GCP_REGION \
#   --display-name=$DISPLAY_NAME \
#   --python-package-uris=$PYTHON_PACKAGE_URI \
#   --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,executor-image-uri=$EXECUTOR_IMAGE_URI,python-module=$PYTHON_MODULE \
#   --args=$CMDARGS

gcloud ai custom-jobs create \
  --region=$GCP_REGION \
  --display-name=$DISPLAY_NAME \
  --python-package-uris=$PYTHON_PACKAGE_URI \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,executor-image-uri=$EXECUTOR_IMAGE_URI,python-module=$PYTHON_MODULE \
  --args=$CMDARGS