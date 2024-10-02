import os
import json
import cv2
from PIL import Image
import numpy as np
from pycocotools.mask import decode

class DSADataset():

    def __init__(self, root, label, resolution, pseudo_label="chendelong/DirectSAM-1800px-0424", **kwargs):

        self.root = root
        self.label = label
        self.resolution = resolution
        self.pseudo_label = pseudo_label

        assert self.label in ['merged', 'pseudo', 'human']

        self.subsets = os.listdir(root)
        self.subsets.sort()
        self.samples = []

        for subset in self.subsets:
            
            # if subset!='SA1B':
            #     continue

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
    
    def get_pseudo_label(self, sample):

        for pseudo_label in sample['pseudo_label']:
            if pseudo_label['source'] == self.pseudo_label:
                return decode(pseudo_label['label'])
        
        raise ValueError(f'Pseudo label {self.pseudo_label} not found in sample {sample}')
    
    def __getitem__(self, idx):

        file = self.samples[idx]
        sample = json.load(open(os.path.join(self.root, file), 'r'))

        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution))

        if self.label == 'pseudo':
            label = self.get_pseudo_label(sample)

        elif self.label == 'human':
            label = self.get_human_labels(sample)

        elif self.label == 'merged':

            pseudo_label = self.get_pseudo_label(sample)
            human_labels = self.get_human_labels(sample)

            label = (pseudo_label + human_labels) > 0
        
        label = cv2.resize(
            label.astype(np.uint8), (self.resolution, self.resolution), 
            interpolation=cv2.INTER_NEAREST
            ) > 0

        return {'image': image, 'label': label, 'label_map': label}