import random
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import cv2
import tqdm

from data.create_dataset import create_dataset

dataset_configs = json.load(open('data/dataset_configs.json', 'r'))
print(dataset_configs.keys())


key = 'cityscapes'


# for key in ['plantorgans','NYUDepthv2', 'VegAnn', 'tcd', 'sidewalk', 'FoodSeg103', 'cityscapes']:
    
print(key)
config = dataset_configs[key]
dataset = create_dataset(config, 'train', 1024, thickness=7)

root = f"/home/dchenbs/workspace/datasets/HF_SEG_Dataset_Images/{key}/train"
os.makedirs(root, exist_ok=True)

for index in tqdm.tqdm(range(len(dataset))):
    sample = dataset[index]
    image = sample['image']
    image_path = os.path.join(root, f"{index}.jpg")
    image.save(image_path)
