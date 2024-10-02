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

	"SA1B"					# 318557
)


cd /home/dchenbs/workspace/DirectSAM
conda activate subobject
for dataset in $datasets; do

    echo -e ">>> All Datasets to Run: $datasets\n\n>>> Current Dataset: $dataset\n"
    
    CUDA_VISIBLE_DEVICES=1 python pseudo_labeling.py \
        --dataset $dataset \
        --checkpoint "chendelong/DirectSAM-1800px-0424" \
        --resolution 1800 --threshold 0.5 --thickness 9 \
        --output_dir "/home/dchenbs/workspace/datasets/DSA/DirectSAM-1800px-0424" \
        --samples 318557
done



```