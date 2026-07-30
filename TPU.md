export PROJECT_ID=raden-trc                   
export TPU_NAME=tpu1                   
export ZONE=europe-west4-b   
export ACCELERATOR_TYPE=v5litepod-1
export RUNTIME_VERSION=v2-alpha-tpuv5-lite

gcloud compute tpus tpu-vm create $TPU_NAME \
 --accelerator-type=$ACCELERATOR_TYPE \
 --version=$RUNTIME_VERSION \
 --zone=$ZONE \
 --project=$PROJECT_ID

 gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE
 gcloud compute tpus tpu-vm ssh tpu1 --zone=europe-west4-b  