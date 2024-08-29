import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from evaluation.metrics import recall_with_tolerance
from data.create_dataset import create_dataset
from model.directsam import DirectSAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_configs', default='data/dataset_configs.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--dataset_name', default='PascalPanopticParts')
    parser.add_argument('--directsam_ckpt', default='chendelong/DirectSAM-tiny-distilled-15ep-768px-0821')
    args = parser.parse_args()

    tolerance = args.resolution // 100 
    tolerance += int(tolerance % 2 == 0)

    dataset_configs = json.load(open(args.dataset_configs, 'r'))
    dataset = create_dataset(
        dataset_configs[args.dataset_name],
        split='validation',
        resolution=args.resolution,
        thickness=2
    )
    if args.n_samples == -1:
        args.n_samples = len(dataset)
    else:
        args.n_samples = min(args.n_samples, len(dataset))

    model = DirectSAM(args.directsam_ckpt, args.resolution, args.threshold, args.device)

    print(args)
    all_num_tokens = []
    all_recall = []
    for i in tqdm.tqdm(range(args.n_samples)):
        sample = dataset[i]

        if type(sample) == dict:
            image = sample['image']
            target = sample['label']
        else:
            image, target = sample

        prediction, num_tokens = model(image)
        recall = recall_with_tolerance(target, prediction, tolerance)

        all_num_tokens.append(num_tokens)
        all_recall.append(recall)

    results = vars(args)
    results['mean_recall'] = np.mean(all_recall)
    results['mean_num_tokens'] = np.mean(all_num_tokens)
    print(results)

    results['all_recall'] = all_recall.tolist()
    results['all_num_tokens'] = all_num_tokens.tolist()

    # {datetime.now().strftime('%m%d_%H%M')}
    output_dir = f"outputs/{args.dataset_name}/{args.directsam_ckpt.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    json.dump(results, open(os.path.join(output_dir, 'results.json'), 'w'), indent=4)