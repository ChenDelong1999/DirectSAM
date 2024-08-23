# DirectSAM Self-Training

Pseudo Labeling


```bash
# single folder
cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
CUDA_VISIBLE_DEVICES=0 python pseudo_labeling.py \
    --image_dir "/home/dchenbs/workspace/datasets/LoveDA/Train/Urban/images_png" \
    --output_dir "/home/dchenbs/workspace/datasets/LoveDA/directsam/Urban/res_1800px_thr.03_output" \
    --checkpoint "chendelong/DirectSAM-1800px-0424" --resolution 1800 --threshold 0.3
```

```bash
# loop over multiple folders

cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
for i in $(seq -f "%06g" 125 149)
do
    echo $i
    CUDA_VISIBLE_DEVICES=5 python pseudo_labeling.py \
        --image_dir "/home/dchenbs/workspace/datasets/sa1b/sa_$i" \
        --output_dir "/home/dchenbs/workspace/datasets/sa1b_directsam/res_1800px_thr.03_output" \
        --checkpoint "chendelong/DirectSAM-1800px-0424" --resolution 1800 --threshold 0.3
done

```


Self-training
```bash
cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port 29510 train.py \
    --pretrained nvidia/segformer-b0-finetuned-cityscapes-1024-1024 \
    --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --dataset directsam_pseudo_label \
    --num_train_epochs 10 \
    --input_resolution 1024 --thickness 3 \
    --dataloader_num_workers 16 --dataloader_prefetch_factor 8
    
```



# Evaluation

## Dataset Preparation

### PartImageNet

https://github.com/TACJu/PartImageNet


### LVIS

https://www.lvisdataset.org/dataset



```bash

ckpts=(
    "chendelong/DirectSAM-1800px-0424"
    # "chendelong/DirectSAM-EntitySeg-1024px-0501"
    # "chendelong/DirectSAM-tiny-distilled-10ep-1024px-0726"
    "chendelong/DirectSAM-tiny-distilled-15ep-768px-0821"
)

datasets=(
    # "PascalPanopticParts"
    "EntitySeg"
    "ADE20k"
    "COCONut_relabeld_COCO_val"
    "LoveDA"
    "PartImageNet++"

    
)


conda activate subobject
for ckpt in $ckpts; do
  for dataset in $datasets; do

    CUDA_VISIBLE_DEVICES=5 python evaluate.py \
        --dataset_name $dataset \
        --directsam_ckpt $ckpt \
        --resolution 1024 --n_samples 1000 \
        --threshold_steps 0.01 --bin_step 32 --max_num_tokens 256 

    done
done

```


```bash

    --directsam_ckpt chendelong/DirectSAM-1800px-0424 \
    --directsam_ckpt chendelong/DirectSAM-EntitySeg-1024px-0501 \
    --directsam_ckpt chendelong/DirectSAM-tiny-distilled-10ep-1024px-0726 \
    --directsam_ckpt chendelong/DirectSAM-tiny-distilled-7ep-768px-0807 \
    --directsam_ckpt chendelong/DirectSAM-tiny-distilled-15ep-768px-0821 \
    --directsam_ckpt /home/dchenbs/workspace/subobjects-dev/DirectSAM/runs/finetune/0822-1154-[directsam_pseudo_label]-768px-from-nvidia_segformer-b0-finetuned-cityscapes-1024-1024/checkpoint-343000 \

```