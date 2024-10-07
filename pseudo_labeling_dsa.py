
import os
import tqdm
import numpy as np
from PIL import Image
import json
from model.directsam import DirectSAM
from pycocotools.mask import encode
import argparse
import datetime



class DSADataset():

    def __init__(self, root):

        self.root = root

        self.subsets = os.listdir(root)
        self.subsets.sort()
        self.samples = []

        for subset in self.subsets:

            files = os.listdir(os.path.join(root, subset))
            files = [x for x in files if x.endswith('.json') and not x.startswith('_')]
            files.sort()
                
            for file in files:
                self.samples.append(os.path.join(subset, file))

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):

        file = self.samples[idx]
        
        return file


dataset_configs = json.load(open('data/dataset_configs.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint")
parser.add_argument("--root", type=str, help="root to DSA")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--resolution", type=int, help="Resolution of DirectSAM model")
parser.add_argument("--threshold", type=float, help="Threshold value for segmentation")
parser.add_argument("--samples", type=int, default=-1, help="Number of samples to use")
parser.add_argument("--thickness", type=int, default=5, help="Thickness of the boundary")

args = parser.parse_args()

model = DirectSAM(args.checkpoint, args.resolution, 'cuda')

os.makedirs(args.output_dir, exist_ok=True)

dataset = DSADataset(args.root)
print(f'Loaded {len(dataset)} samples')

args.samples = len(dataset) if args.samples == -1 else args.samples
args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M")
print(args)

index_mapping = np.random.permutation(min(args.samples, len(dataset)))
for i in tqdm.tqdm(range(args.samples)):

    i = int(index_mapping[i])
    file  = dataset[i]

    subset, basename = file.split('/')

    os.makedirs(os.path.join(args.output_dir, subset), exist_ok=True)

    output_file = os.path.join(args.output_dir, file)
    if os.path.exists(output_file):
        continue

    try: 

        sample = json.load(open(os.path.join(args.root, file), 'r'))
        image = Image.open(sample['image_path']).convert('RGB')
        
        probabilities = model(image)
        pseudo_label = probabilities > args.threshold

        rle_prediction = encode(np.array(pseudo_label, order='F', dtype=np.uint8))
        rle_prediction['counts'] = rle_prediction['counts'].decode('utf-8')

        sample['pseudo_label'].append(
            {
                'source': args.checkpoint,
                'label': rle_prediction
            }
        )

        # remove "info" from sample
        if 'info' in sample:
            sample.pop('info')

        with open(output_file, 'w') as f:
            json.dump(sample, f, indent=4)

    except Exception as e:

        print(f'Error: {e}')
        continue
