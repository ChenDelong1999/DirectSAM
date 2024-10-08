import os
import tqdm
import numpy as np
from PIL import Image
import json
from pycocotools.mask import encode, decode
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--resolution", type=int, help="Resolution of DirectSAM model")
parser.add_argument("--samples", type=int, default=-1, help="Number of samples to use")
parser.add_argument("--thickness", type=int, default=5, help="Thickness of the boundary")

args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir)
os.makedirs(os.path.join(args.output_dir, 'SA1B'), exist_ok=True)


class SA1BDataset:

    def __init__(self, root, thickness=5, max_num_masks=512):
        self.image_paths = []
        for subfolder in tqdm.tqdm(os.listdir(root)):
            subfolder_path = os.path.join(root, subfolder)
            if os.path.isdir(subfolder_path):
                for img_file in os.listdir(subfolder_path):
                    if img_file.endswith('.jpg'):
                        image_path = os.path.join(subfolder_path, img_file)
                        json_path = image_path.replace('.jpg', '.json')
                        if os.path.exists(json_path):
                            self.image_paths.append(image_path)

        print("Total images:", len(self.image_paths))
        print("First few image paths:", self.image_paths[:5])

        # Device (use 'cuda' if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Thickness and maximum number of masks
        self.thickness = thickness
        self.max_num_masks = max_num_masks

        # Create the kernel once
        self.kernel = self.create_circular_kernel_torch(self.thickness).to(self.device)
        self.kernel_sum = self.kernel.sum()

        # Initialize masks placeholder (will be dynamically resized as needed)
        self.masks = None
        self.current_mask_shape = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]

            json_path = image_path.replace('.jpg', '.json')
            with open(json_path, 'r') as f:
                annotations = json.load(f)['annotations']

            # Limit the number of annotations to max_num_masks
            annotations = annotations[:self.max_num_masks]

            # Create the label map
            label_map = None
            for i, annotation in enumerate(annotations):
                segmentation = decode(annotation['segmentation'])
                if label_map is None:
                    label_map = np.zeros_like(segmentation, dtype=np.int32)
                label = (i + 1)
                label_map += segmentation * label

            # Compute the boundary
            boundary = self.label_map_to_boundary_torch(label_map)

            return image_path, boundary

        except Exception as e:
            print(f"Error with image at index {idx} ({self.image_paths[idx]}):\n{e}")
            # Skip this image and proceed to the next one
            return None, None

    def create_circular_kernel_torch(self, size):
        if size > 3:
            kernel = torch.zeros((size, size), dtype=torch.float32)
            center = size // 2
            y, x = torch.meshgrid(
                torch.arange(0, size, dtype=torch.float32),
                torch.arange(0, size, dtype=torch.float32),
                indexing='ij'
            )
            dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
            kernel[dist <= center] = 1.0
        else:
            kernel = torch.ones((size, size), dtype=torch.float32)
        return kernel

    def label_map_to_boundary_torch(self, label_map):
        # Convert label_map to PyTorch tensor
        label_map = torch.from_numpy(label_map).to(self.device)

        unique_labels = torch.unique(label_map)
        unique_labels = unique_labels[unique_labels != 0]
        num_labels = unique_labels.numel()

        if num_labels == 0:
            # If there are no labels, return a boundary map with borders
            boundary = torch.zeros_like(label_map, dtype=torch.bool)
            boundary[:self.thickness, :] = True
            boundary[-self.thickness:, :] = True
            boundary[:, :self.thickness] = True
            boundary[:, -self.thickness:] = True
            return boundary.cpu().numpy()

        # Prepare masks tensor
        labels_shape = label_map.shape
        if self.masks is None or self.current_mask_shape != labels_shape:
            # Initialize the masks tensor if not already done or if shape has changed
            self.masks = torch.zeros(
                (1, self.max_num_masks, *labels_shape),
                dtype=torch.float32,
                device=self.device
            )
            self.current_mask_shape = labels_shape

        # Create masks for each label
        num_labels = min(num_labels, self.max_num_masks)
        labels = unique_labels[:num_labels].view(-1, 1, 1)  # Shape: (num_labels, 1, 1)
        masks = (label_map == labels).float()  # Shape: (num_labels, H, W)
        masks = masks.unsqueeze(0)  # Shape: (1, num_labels, H, W)

        # Adjust masks tensor
        self.masks[:, :num_labels, :, :] = masks
        masks = self.masks[:, :num_labels, :, :]

        # Prepare the kernel
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kH, kW)
        kernel = kernel.expand(num_labels, 1, *self.kernel.shape)  # Shape: (num_labels, 1, kH, kW)

        padding = self.thickness // 2

        # Perform dilation and erosion using depthwise convolution
        dilated = torch.nn.functional.conv2d(
            masks, kernel, padding=padding, groups=num_labels
        ) > 0

        eroded = torch.nn.functional.conv2d(
            masks, kernel, padding=padding, groups=num_labels
        ) == self.kernel_sum

        # Compute boundaries
        boundaries = dilated & (~eroded)

        # Combine boundaries from all labels
        boundary = boundaries.any(dim=1).squeeze(0)  # Shape: (H, W)

        # Include image borders in the boundary
        boundary[:self.thickness, :] = True
        boundary[-self.thickness:, :] = True
        boundary[:, :self.thickness] = True
        boundary[:, -self.thickness:] = True

        # Convert to NumPy array
        boundary = boundary.cpu().numpy().astype(bool)

        return boundary


if __name__ == '__main__':

    # Load dataset configurations
    dataset_configs = json.load(open('data/dataset_configs.json', 'r'))
    dataset_root = dataset_configs['SA1B']['root']
    dataset = SA1BDataset(dataset_root, thickness=args.thickness)

    args.samples = len(dataset) if args.samples == -1 else args.samples
    print("Arguments:", args)

    index_mapping = np.random.permutation(min(args.samples, len(dataset)))
    for idx in tqdm.tqdm(range(args.samples)):
        i = int(index_mapping[idx])
        output_file = os.path.join(args.output_dir, 'SA1B', f'{i:07d}.json')
        if os.path.exists(output_file):
            continue

        image_path, label = dataset[i]

        if image_path is None or label is None:
            continue  # Skip if there was an error with this image

        try:
            rel_human_label = encode(np.array(label, order='F', dtype=np.uint8))
            rel_human_label['counts'] = rel_human_label['counts'].decode('utf-8')

            result = {
                'image_path': image_path,
                'human_label': [{
                    'source': 'SA1B',
                    'label': rel_human_label
                }]
            }

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)

        except Exception as e:
            print(f"Error processing image {image_path}:\n{e}")