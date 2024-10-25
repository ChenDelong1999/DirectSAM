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
cd /cpfs/shared/research-llm/liujianfeng/08_subobject/DirectSAM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=1 python sa1b_mask_to_contour.py \
    --resolution 1024 --thickness 5 \
    --output_dir "/cpfs/shared/research-llm/liujianfeng/08_subobject/data/sa1b_contour" \
    --samples -1
```


```bash
# On CPFS
cd /cpfs/shared/research-llm/liujianfeng/08_subobject/DirectSAM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29511 train.py \
    --pretrained "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --dataset SA1B \
    --num_train_epochs 10 \
    --input_resolution 1024 --thickness 5 \
    --dataloader_num_workers 32 --dataloader_prefetch_factor 16
```

### Stage 2: Generating and Merging Pseudo-labels

```bash

datasets=(
	# "COCONut-b"  	        # 123403
	# "COCONut-s"  	        # 118287
	# "LVIS"  	            # 100170
	# "PartImageNetPP"  	    # 90000
	# "COCONut-l"  	        # 63256
	# "PACO"  	            # 45790
	# "Fashionpedia"  	    # 45623
	# "EntitySeg"  	        # 31913
	# "LIP"  	                # 30462
	# "CelebA"  	            # 30000
	# "CIHP"  	            # 28280
	# "ADE20k"  	            # 20210
	# "SeginW"  	            # 10639
	# "DUTS-TR"  	            # 10553
	# "fss_all"  	            # 10000
	# "MSRA_10K"  	        # 10000
	# "GTA5"  	            # 10000
	# "SPIN"  	            # 8828
	# "PascalPanopticParts"  	# 4998
	# "ThinObject5K"  	    # 4748
	# "DIS5K-DIS-TR"  	    # 3000
	# "LoveDA"  	            # 2522
	# "SUIM"  	            # 1221
	# "ecssd"  	            # 1000
	# "SOBA"  	            # 933
	# "MyFood"  	            # 750
	# "COIFT"  	            # 280
	# "HRSOD"  	            # 287
	# "plantorgans"			# 5745
	# "FoodSeg103"			# 4983
	# "tcd"					# 4169
	# "VegAnn"				# 3775
	# "cityscapes"			# 2975
	# "sidewalk"				# 1000
	# "NYUDepthv2"			# 795
	# "UAVID"					# 200
	# "EgoHOS"				# 8993
	# "PhenoBench"			# 1407
	# "TreeCount"				# 83
	# "SA1B"					# 318557
    # "OpenEarthMap"            # 10748
    # "WireFrame"           # 5000
    "ISAID"               # 4792
)


cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
for dataset in $datasets; do

    echo -e ">>> All Datasets to Run: $datasets\n\n>>> Current Dataset: $dataset\n"
    
    CUDA_VISIBLE_DEVICES=2 python pseudo_labeling.py \
        --dataset $dataset \
        --checkpoint "chendelong/DirectSAM-1800px-0424" \
        --resolution 1800 --threshold 0.5 --thickness 9 \
        --output_dir "/home/dchenbs/workspace/datasets/DSA/DirectSAM-1800px-0424" \
        --samples -1
done

```


### Stage 3: Weakly Supervised Sefl-Training on DSA

```bash

export NCCL_P2P_LEVEL=NVL

cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29512 train.py \
    --pretrained "chendelong/DirectSAM-b0-1024px-sa1b-2ep-1016" \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --dataset DSA_gen3 \
    --num_train_epochs 100 \
    --input_resolution 1024 \
    --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
    --do_eval #--fp16
    
```

### Stage 4: Adding new generation of Pseudo-labels

```bash
cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
CUDA_VISIBLE_DEVICES=5 python pseudo_labeling_dsa.py \
    --root "/home/dchenbs/workspace/datasets/DSA/DirectSAM-gen2-1024px-1014" \
    --output_dir "/home/dchenbs/workspace/datasets/DSA/DirectSAM-gen3-1024px-1023" \
    --checkpoint "chendelong/DirectSAM-gen3-1024px-1023" \
    --resolution 1024 --threshold 0.5 --thickness 5 \
    --samples -1



    # --root /home/dchenbs/workspace/datasets/DSA/DirectSAM-1800px-0424 \
    # --output_dir "/home/dchenbs/workspace/datasets/DSA/DirectSAM-gen1-1024px-1008" \

    # --root "/home/dchenbs/workspace/datasets/DSA/DirectSAM-gen1-1024px-1008" \
    # --output_dir "/home/dchenbs/workspace/datasets/DSA/DirectSAM-gen2-1024px-1014" \
```

### Evaluation



```bash

thresholds=(
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
)

ckpts=(
    # "chendelong/DirectSAM-1800px-0424"
    # "chendelong/DirectSAM-gen1-1024px-1008"
    # "chendelong/DirectSAM-gen2-1024px-1014"
    # "chendelong/DirectSAM-b0-1024px-sa1b-2ep-1016"
    "chendelong/DirectSAM-gen3-1024px-1023"
)

datasets=(
    "EntitySeg"
    "PascalPanopticParts"
    "SA1B_116"
)

cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
for ckpt in $ckpts; do
    for dataset in $datasets; do
        for threshold in $thresholds; do

            CUDA_VISIBLE_DEVICES=5 python evaluate.py \
                --dataset_name $dataset \
                --directsam_ckpt $ckpt \
                --resolution 1024 \
                --tolerance 10 \
                --n_samples 1000 \
                --threshold $threshold \
                --output_dir "outputs/effective_boundary_recall" 
                # --sleep_interval 0.1
        done
    done
done

```



### Push to hub

```python
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

checkpoint = "/home/dchenbs/workspace/DirectSAM/runs/DSA_gen2/1018-0426-1024px-from-chendelong_DirectSAM-gen2-1024px-1014/checkpoint-140000"
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
model.push_to_hub("chendelong/DirectSAM-gen3-1024px-1023")

```