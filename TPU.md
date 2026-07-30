```
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
```

```
gcloud compute tpus tpu-vm scp ~/.ssh/id_ed25519  ~/.ssh/id_ed25519.pub tpu1:~/.ssh/ --zone=europe-west4-b

chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
ssh -o StrictHostKeyChecking=accept-new -T git@github.com

gcloud compute tpus tpu-vm ssh tpu1 --zone=europe-west4-b
git clone (https://github.com/radenmuaz/kvmem)
cd kvmem
# work
```

```
32 on-demand Cloud TPU v4 chips in zone us-central2-b
32 spot Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v6e chips in zone us-east1-d
64 spot Cloud TPU v6e chips in zone europe-west4-a
64 spot Cloud TPU v5e chips in zone europe-west4-b
64 spot Cloud TPU v5e chips in zone us-central1-a
 ```