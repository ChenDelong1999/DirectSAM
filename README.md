### DirectSAM Pseudo Labeling on SA-1B


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

### Pseudo Labeling and annotation merging on other datasets

```bash

datasets=(
    # "LIP"
    # "CelebA"
    # "SOBA"
    # "SeginW"

    # "CIHP"
    # "Fashionpedia"
    # "PascalPanopticParts"
    # "SPIN"

    # "PartImageNet++"
    # "ADE20k"
    # "EntitySeg"
    # "LoveDA"

    # "COCONut-s"
    # "COCONut-b"
    "COCONut-l"

    # "PACO"
    # "LVIS"
    # "COIFT"
    # "DIS5K-DIS-TR"
    # "DUTS-TR"

    # "ecssd"
    # "fss_all"
    # "HRSOD"
    # "MSRA_10K"
    # "ThinObject5K"
)


cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
for dataset in $datasets; do

    echo -e ">>> All Datasets to Run: $datasets\n\n>>> Current Dataset: $dataset\n"
    
    CUDA_VISIBLE_DEVICES=5 python pseudo_labeling_merge.py \
        --checkpoint "chendelong/DirectSAM-1800px-0424" \
        --resolution 1800 --threshold 0.5 \
        --dataset $dataset --thickness 3 \
        --output_dir "/home/dchenbs/workspace/datasets/DirectSAPlus/DirectSAM-1800px-0424" \
        --samples -1
done


    # --do_post_processing \
```


### Self-training

```bash

export NCCL_P2P_LEVEL=NVL

cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node 2 --master_port 29510 train.py \
    --pretrained "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 64 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --dataset directsam_pseudo_label_merged_denoised \
    --num_train_epochs 100 \
    --input_resolution 768 --thickness 3 \
    --dataloader_num_workers 16 --dataloader_prefetch_factor 8
    
```

```bash
    --pretrained "chendelong/DirectSAM-tiny-distilled-30ep-1024px-0906" \

    --dataset directsa_plus \

    --dataset directsa_plus \

# 1024px input resolution

    # 3.7M Parameters, 75.3G CUDA Memory, ~7 hours/epoch with 2 GPU
    --pretrained "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \

    # 13.6M Parameters, 49.3G CUDA Memory, ~8 hours/epoch with 2 GPU
    --pretrained "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 2 \

    # 27.3M Parameters, 45.7G CUDA Memory, ~13 hours/epoch with 2 GPU
    --pretrained "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \

    # 47.2M Parameters, 60.3G CUDA Memory, ~17 hours/epoch with 2 GPU
    --pretrained "nvidia/segformer-b3-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \

    # 64.0M Parameters, 43.3G CUDA Memory, ~25 hours/epoch with 2 GPU
    --pretrained "nvidia/segformer-b4-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \

    # 84.6M Parameters, 49.7G CUDA Memory, ~28 hours/epoch with 2 GPU
    --pretrained "nvidia/segformer-b5-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \

    --pretrained "chendelong/DirectSAM-1800px-0424" \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \

```


# Evaluation

## Dataset Preparation

### PartImageNet

https://github.com/TACJu/PartImageNet


### LVIS

https://www.lvisdataset.org/dataset



```bash

thresholds=(
    0.05
    0.1
    0.15
    0.2
    0.25
    0.3
    0.35
    0.4
    0.45
    0.5
    # 0.6
    # 0.7
    # 0.8
    # 0.9
)

ckpts=(
    # "chendelong/DirectSAM-1800px-0424"
    # "chendelong/DirectSAM-EntitySeg-1024px-0501"
    # "chendelong/DirectSAM-tiny-distilled-10ep-1024px-0726"
    # "chendelong/DirectSAM-tiny-distilled-15ep-768px-0821"
    # "runs/directsam_pseudo_label_merged/0828-2024-1024px-from-chendelong_DirectSAM-1800px-0424/checkpoint-28000"

    # "chendelong/DirectSAM-tiny-distilled-30ep-1024px-0906"
    # "chendelong/DirectSAM-tiny-distilled-30ep-plus-30ep-1024px-0907"
    # "chendelong/DirectSAM-tiny-distilled-30ep-plus-50ep-1024px-0910"
    "chendelong/DirectSAM-tiny-distilled-70ep-1024px-0920"
)

datasets=(
    # "PascalPanopticParts"
    # "ADE20k"
    # "COCONut_relabeld_COCO_val"
    "EntitySeg"

    # "LIP"
    # "DRAM"
    # "SOBA"
    # "SeginW"
    # "CIHP"
    # "Fashionpedia"
    # "SPIN"

    # "PartImageNet++"
    # "LoveDA"
    # "PACO"
    # "DIS5K-DIS-VD"
    # "DUTS-TE"
)

cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
for ckpt in $ckpts; do
    for dataset in $datasets; do
        for threshold in $thresholds; do

            CUDA_VISIBLE_DEVICES=3 python evaluate.py \
                --dataset_name $dataset \
                --directsam_ckpt $ckpt \
                --resolution 1024 \
                --n_samples 1000 \
                --threshold $threshold \
                --output_dir "outputs/eval_token_recall" --sleep_interval 0.2

        done
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



```python
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

checkpoint = "/home/dchenbs/workspace/DirectSAM/runs/directsam_pseudo_label_merged/0910-2233-1024px-from-nvidia_segformer-b0-finetuned-cityscapes-1024-1024/checkpoint-240000"
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
model.push_to_hub("chendelong/DirectSAM-tiny-distilled-70ep-1024px-0920")

```