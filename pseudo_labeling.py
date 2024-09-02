
import os
import tqdm
import numpy as np
from PIL import Image
import json
from model.directsam import DirectSAM
from pycocotools.mask import encode
import argparse
import datetime
from data.create_dataset import create_dataset

dataset_configs = json.load(open('data/dataset_configs.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint")
parser.add_argument("--dataset", type=str, help="ID of the dataset")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--resolution", type=int, help="Resolution of DirectSAM model")
parser.add_argument("--threshold", type=float, help="Threshold value for segmentation")
parser.add_argument("--samples", type=int, default=-1, help="Number of samples to use")
parser.add_argument("--thickness", type=int, default=5, help="Thickness of the boundary")
parser.add_argument("--do_post_processing", action='store_true', help="Whether to do post processing")

args = parser.parse_args()

model = DirectSAM(args.checkpoint, args.resolution, args.threshold, 'cuda')

args.output_dir = os.path.join(args.output_dir, os.path.basename(args.dataset))
os.makedirs(args.output_dir, exist_ok=True)

dataset = create_dataset(dataset_configs[args.dataset], split='train', resolution=args.resolution, thickness=args.thickness)

args.samples = len(dataset) if args.samples == -1 else args.samples
args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M")
print(args)

index_mapping = np.random.permutation(len(dataset))
for i in tqdm.tqdm(range(args.samples)):

    i = int(index_mapping[i])

    output_file = os.path.join(args.output_dir, f'{i:07d}.json')
    if os.path.exists(output_file):
        continue

    try: 
        image_path = dataset.image_paths[i]
        image_size = Image.open(image_path).size

        sample = dataset[i]
        if type(sample) == dict:
            image = sample['image']
            target = sample['label']
        else:
            image, target = sample

        prediction, num_tokens = model(image, post_processing=args.do_post_processing)
        merged = (prediction + target) > 0

        rle_prediction = encode(np.array(prediction, order='F', dtype=np.uint8))
        rle_prediction['counts'] = rle_prediction['counts'].decode('utf-8')

        rel_target = encode(np.array(target, order='F', dtype=np.uint8))
        rel_target['counts'] = rel_target['counts'].decode('utf-8')

        rle_merged = encode(np.array(merged, order='F', dtype=np.uint8))
        rle_merged['counts'] = rle_merged['counts'].decode('utf-8')

        result = {}
        result['image_path'] = image_path
        result['image_size'] = image_size
        result['info'] = vars(args)
        result['prediction'] = rle_prediction
        result['target'] = rel_target
        result['merged'] = rle_merged

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

    except Exception as e:

        print(f'Error: {e}')
        continue
