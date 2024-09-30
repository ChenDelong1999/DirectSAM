# Direct Segment Anything Model (DirectSAM)

## Data Preparation

- [SA-1B](https://ai.meta.com/datasets/segment-anything/): download ([OpenDataLab](https://opendatalab.com/OpenDataLab/SA-1B)) and extract.





## Install Dependencies

```bash
conda create -n directsam python=3.8

```


## Training

### Stage 1: Large-scale Pretraining on SA-1B

```bash

cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 29511 train.py \
    --pretrained "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --dataset SA1B \
    --num_train_epochs 100 \
    --input_resolution 1800 --thickness 3 \
    --dataloader_num_workers 16 --dataloader_prefetch_factor 8
    
```

### Stage 2: Generating and Merging Pseudo-labels

```bash


```