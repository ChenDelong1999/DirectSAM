import os
import tqdm
import numpy as np
from PIL import Image
import json
from pycocotools.mask import encode
import argparse
import datetime
from data.create_dataset import create_dataset
import multiprocessing
import functools

dataset_configs = json.load(open('data/dataset_configs.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--resolution", type=int, help="Resolution of DirectSAM model")
parser.add_argument("--samples", type=int, default=-1, help="Number of samples to use")
parser.add_argument("--thickness", type=int, default=5, help="Thickness of the boundary")

args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir)
os.makedirs(os.path.join(args.output_dir, 'SA1B'), exist_ok=True)

# Global dataset variable
dataset = create_dataset(dataset_configs['SA1B'], split='train', resolution=args.resolution, thickness=args.thickness)

args.samples = len(dataset) if args.samples == -1 else args.samples
args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M")
print(args)

index_mapping = np.random.permutation(min(args.samples, len(dataset)))

def process_sample(i, args, index_mapping):
    i = int(index_mapping[i])
    output_file = os.path.join(args.output_dir, 'SA1B', f'{i:07d}.json')
    if os.path.exists(output_file):
        return

    try: 
        image_path = dataset.image_paths[i]
        # image_size = Image.open(image_path).size

        sample = dataset[i]
        # image = sample['image']
        human_label = sample['label']

        rel_human_label = encode(np.array(human_label, order='F', dtype=np.uint8))
        rel_human_label['counts'] = rel_human_label['counts'].decode('utf-8')

        result = {}
        result['image_path'] = image_path
        # result['image_size'] = image_size
        result['info'] = vars(args)
        result['human_label'] = [{
            'source': 'SA1B',
            'label': rel_human_label
            }]

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        list(tqdm.tqdm(pool.imap(functools.partial(process_sample, args=args, index_mapping=index_mapping), range(args.samples)), total=args.samples))