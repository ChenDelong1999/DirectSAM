import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import cv2

from evaluation.metrics import effective_contour_recall
from evaluation.visualization import compare_boundaries
from data.create_dataset import create_dataset
from model.directsam import DirectSAM

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_configs', default='data/dataset_configs.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--tolerance', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--dataset_name', default='PascalPanopticParts')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--sleep_interval', type=float, default=0)
    parser.add_argument('--directsam_ckpt', default='chendelong/DirectSAM-tiny-distilled-15ep-768px-0821')
    args = parser.parse_args()

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

    model = DirectSAM(args.directsam_ckpt, args.resolution, args.device)

    # {datetime.now().strftime('%m%d_%H%M')}
    output_dir = f"{args.output_dir}/{args.dataset_name}/{'-'.join(args.directsam_ckpt.split('/')[-2:])}/threshold@{args.threshold}"
    os.makedirs(output_dir, exist_ok=True)
    date_time = datetime.now().strftime("%m%d-%H%M")

    print(args)
    all_num_tokens = []
    all_recall = []
    for i in tqdm.tqdm(range(args.n_samples)):
        if args.sleep_interval > 0:
            time.sleep(args.sleep_interval)
        sample = dataset[i]

        image = sample['image']
        target = sample['label']

        probs = model(image)
        prediction = probs > args.threshold

        num_tokens, recall = effective_contour_recall(target, prediction, args.tolerance)
        
        if recall==-1:
            continue

        all_num_tokens.append(num_tokens)
        all_recall.append(recall)

        if i<5:

            plt.figure(figsize=(30, 6))

            num_tokens, recall = effective_contour_recall(target, prediction, args.tolerance, do_visualization=True)

            plt.subplot(1, 5, 1)
            plt.title('Input Image')
            plt.imshow(image)
            plt.axis('off')

            for subplot in [2,3,4]:
                plt.subplot(1, 5, subplot)
                plt.imshow(image, alpha=0.3)

            plt.savefig(os.path.join(output_dir, f'{i:07d}.png'), bbox_inches='tight')

    results = vars(args)
    results['mean_recall'] = np.mean(all_recall)
    results['mean_num_tokens'] = np.mean(all_num_tokens)

    df = pd.DataFrame.from_dict(results, orient='index')
    # transpose this df
    df = df.T
    df.to_csv(os.path.join(output_dir, f'results-{date_time}.csv'), sep='\t', index=False)
    print(df)

    results['all_recall'] = all_recall
    results['all_num_tokens'] = all_num_tokens

    json.dump(results, open(os.path.join(output_dir, f'results-{date_time}.json'), 'w'), indent=4)