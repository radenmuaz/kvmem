Do not use TPU API, must use `gcloud compute tpus queued-resources` else will be billed not using TRC quota.

Do not request other than these:
```
32 on-demand Cloud TPU v4 chips in zone us-central2-b
32 spot Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v6e chips in zone us-east1-d
64 spot Cloud TPU v6e chips in zone europe-west4-a
64 spot Cloud TPU v5e chips in zone europe-west4-b
64 spot Cloud TPU v5e chips in zone us-central1-a
```

```
gcloud compute tpus tpu-vm ssh tpu2 --zone=europe-west4-a
gcloud compute tpus tpu-vm ssh tpu3 --zone=us-east1-d
```

bigger
```
gcloud compute tpus queued-resources create res2 \
    --node-id tpu2 \
    --project raden-trc \
    --zone europe-west4-a \
    --accelerator-type v6e-1 \
    --runtime-version v2-alpha-tpuv6e \
    --spot

gcloud compute tpus queued-resources create res28 \
    --node-id tpu28 \
    --project raden-trc \
    --zone us-east1-d \
    --accelerator-type v6e-8 \
    --runtime-version v2-alpha-tpuv6e \
    --spot


gcloud compute tpus queued-resources list --project raden-trc --zone europe-west4-a

gcloud compute tpus queued-resources list --project raden-trc --zone us-east1-d

gcloud compute tpus queued-resources delete res2 \
    --project raden-trc \
    --zone europe-west4-a \
    --force \
    --async

gcloud compute tpus queued-resources delete res3 \
    --project raden-trc \
    --zone us-east1-d \
    --force \
    --async
```

```
gcloud compute tpus tpu-vm scp ~/.ssh/id_ed25519  ~/.ssh/id_ed25519.pub tpu1:~/.ssh/ --zone=europe-west4-b

chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
ssh -o StrictHostKeyChecking=accept-new -T git@github.com

git clone (https://github.com/radenmuaz/kvmem)
cd kvmem
# work
```


smaller
```
gcloud compute tpus queued-resources create res1 \
    --node-id tpu1 \
    --project raden-trc \
    --zone us-central1-a \
    --accelerator-type v5litepod-1 \
    --runtime-version v2-alpha-tpuv5-lite \
    --spot

gcloud compute tpus queued-resources create resv5 \
    --node-id tpu3 \
    --project raden-trc \
    --zone us-central2-b \
    --accelerator-type v4-1 \
    --runtime-version tpu-ubuntu2204-base

gcloud compute tpus queued-resources list --project raden-trc --zone us-central1-a

gcloud compute tpus queued-resources describe res1 \
    --project raden-trc \
    --zone us-central1-a

gcloud compute tpus queued-resources delete res1 \
    --project raden-trc \
    --zone us-central1-a \
    --force \
    --async
```