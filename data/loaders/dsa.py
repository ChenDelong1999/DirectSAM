import os
import json
import cv2
from PIL import Image
import numpy as np
from pycocotools.mask import decode

class DSADataset():

    def __init__(self, root, label, resolution, **kwargs):

        self.root = root
        self.label = label
        self.resolution = resolution

        assert self.label in ['merged', 'pseudo', 'human']

        self.subsets = os.listdir(root)
        self.subsets.sort()
        self.samples = []

        for subset in self.subsets:
            files = os.listdir(os.path.join(root, subset))
            files = [x for x in files if x.endswith('.json') and not x.startswith('_')]
            print(f'\tDSA subset {len(files)} samples:\t{subset}')
            for file in files:
                self.samples.append(os.path.join(subset, file))

        self.image_paths = self.samples

    def __len__(self):
        return len(self.samples)
    
    def get_human_labels(self, sample):
    
        if len(sample['human_label']) == 1:
            label = decode(sample['human_label'][0]['label'])
        else:
            human_labels = []
            for human_label in sample['human_label']:
                human_labels.append(decode(human_label['label']))
            label = np.sum(np.array(human_labels), axis=0)

        return label
    
    def __getitem__(self, idx):

        file = self.samples[idx]
        sample = json.load(open(os.path.join(self.root, file), 'r'))

        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution))

        if self.label == 'pseudo':
            label = decode(sample['pseudo_label'])

        elif self.label == 'human':
            label = self.get_human_labels(sample)

        elif self.label == 'merged':

            pseudo_label = decode(sample['pseudo_label'])
            human_labels = self.get_human_labels(sample)

            label = (pseudo_label + human_labels) > 0
        
        label = cv2.resize(
            label.astype(np.uint8), (self.resolution, self.resolution), 
            interpolation=cv2.INTER_NEAREST
            ) > 0

        return {'image': image, 'label': label, 'label_map': label}