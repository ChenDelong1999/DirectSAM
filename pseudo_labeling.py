
import os
import numpy as np
import tqdm
import numpy as np
from PIL import Image
import tqdm
import numpy as np
from PIL import Image
import json

from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from model.directsam import DirectSAM
from pycocotools.mask import encode
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint")
parser.add_argument("--image_dir", type=str, help="Path to the image directory")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--resolution", type=int, help="Resolution of the output")
parser.add_argument("--threshold", type=float, help="Threshold value for segmentation")

args = parser.parse_args()

model = DirectSAM(args.checkpoint, args.resolution, 'cuda')

args.output_dir = os.path.join(args.output_dir, os.path.basename(args.image_dir))
os.makedirs(args.output_dir, exist_ok=True)

image_paths = []
for file in os.listdir(args.image_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_paths.append(os.path.join(args.image_dir, file))

print(args)

for i in tqdm.tqdm(range(len(image_paths))):
    try:
        image = Image.open(image_paths[i]).convert('RGB')
        probs = model(image)
        result = probs > args.threshold

        rle = encode(np.array(result, order='F', dtype=np.uint8))
        rle['counts'] = rle['counts'].decode('utf-8')

        with open(os.path.join(args.output_dir, os.path.basename(image_paths[i]).replace('.jpg', '.json')), 'w') as f:
            json.dump(rle, f, indent=4)
    except Exception as e:
        print(e)
        print(image_paths[i])
        continue
        