import os
from PIL import Image
import numpy as np


class CelebADataset:

    def __init__(self, root):

        self.root = root
        self.annotations = {i: [] for i in range(30000)}

        for subfolder in range(15):
            for file in os.listdir(os.path.join(root, f'CelebAMask-HQ-mask-anno/{subfolder}')):
                if file.endswith('.png'):
                    idx = int(file.split('_')[0])
                    self.annotations[idx].append(os.path.join(root, f'CelebAMask-HQ-mask-anno/{subfolder}', file))

        self.image_paths = [os.path.join(root, f'CelebA-HQ-img/{i}.jpg') for i in range(30000)]

    def __len__(self):
        return 30000
    
    def __getitem__(self, idx):
                
        image = Image.open(os.path.join(self.root, f'CelebA-HQ-img/{idx}.jpg')).convert('RGB')

        label_map = None
        for i, label_path in enumerate(self.annotations[idx]):
            label = np.array(Image.open(label_path)) > 0
            if label_map is None:
                label_map = label * (i + 1)
            else:
                label_map += label * (i + 1)

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed


    def set_transform(self, transform):
        self.transform = transform
    