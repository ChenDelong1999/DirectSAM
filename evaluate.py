import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from evaluation.metrics import get_metrics
from data.create_dataset import create_dataset
from model.directsam import DirectSAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_configs', default='data/dataset_configs.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=768)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--threshold_steps', type=float, default=0.01)
    parser.add_argument('--bin_step', type=int, default=32)
    parser.add_argument('--max_num_tokens', type=int, default=256)
    parser.add_argument('--dataset_name', default='PascalPanopticParts')
    parser.add_argument('--directsam_ckpt', default='chendelong/DirectSAM-tiny-distilled-15ep-768px-0821')
    args = parser.parse_args()

    bzp_offset = args.resolution // 100
    tolerance = args.resolution // 100 + args.resolution % 2
    thresholds = np.linspace(args.threshold_steps, 1 - args.threshold_steps, int(1 / args.threshold_steps) - 1)

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

    all_num_tokens = []
    all_recall = []

    print(args)
    for i in tqdm.tqdm(range(args.n_samples)):
        sample = dataset[i]

        if type(sample) == dict:
            image = sample['image']
            target = sample['label']
        else:
            image, target = sample

        prob = model(image)

        num_tokens, recall = get_metrics(
            target, prob, thresholds,
            bzp_offset=bzp_offset, tolerance=tolerance,
            device=args.device
        )

        all_num_tokens.append(num_tokens)
        all_recall.append(recall)

    y = np.array(all_recall).flatten()
    x = np.array(all_num_tokens).flatten()

    df = pd.DataFrame({'num_tokens': x, 'recall': y})
    bins = np.arange(0, args.max_num_tokens + args.bin_step, args.bin_step)
    df['binned_tokens'] = pd.cut(df['num_tokens'], bins)
    grouped = df.groupby('binned_tokens')['recall'].agg(['mean', 'std']).reset_index()
    new_row = pd.DataFrame({'binned_tokens': ['all'], 'mean': [y.mean()], 'std': [y.std()]})
    grouped = pd.concat([grouped, new_row], ignore_index=True)

    # {datetime.now().strftime('%m%d_%H%M')}
    output_dir = f"outputs/{args.dataset_name}_{args.directsam_ckpt.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    grouped.to_csv(os.path.join(output_dir, 'results.csv'), index=False, sep='\t')

    json.dump(vars(args), open(os.path.join(output_dir, 'args.json'), 'w'), indent=4)