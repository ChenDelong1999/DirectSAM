import os
import json
import cv2
from PIL import Image
import numpy as np
from pycocotools.mask import decode
from torchvision import transforms
import torch

augmentation = transforms.RandomApply(torch.nn.ModuleList([
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomGrayscale(p=0.2),
    ]), p=0.25)


class DSADataset():

    def __init__(self, root, label, resolution, split, pseudo_label="chendelong/DirectSAM-1800px-0424", do_augmentation=False, **kwargs):

        self.root = root
        self.label = label
        self.resolution = resolution
        self.pseudo_label = pseudo_label
        self.split = split
        self.do_augmentation = do_augmentation

        assert self.label in ['merged', 'pseudo', 'human']

        self.subsets = os.listdir(root)
        self.subsets.sort()
        self.samples = []

        for subset in self.subsets:
            
            # if subset!='OpenEarthMap':
            #     continue

            files = os.listdir(os.path.join(root, subset))
            files = [x for x in files if x.endswith('.json') and not x.startswith('_')]
            files.sort()

            if split == 'train':
                files = files[:int(0.99*len(files))]
            elif split == 'validation':
                files = files[int(0.99*len(files)):]

            print(f'\tDSA {split} subset {len(files)} samples:\t{subset}')
                
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
        try:
            return self.getitem(idx)
        except Exception as e:
            print(f'Error in sample {idx}: {e}')
            new_idx = (idx + 1) % len(self)
            return self.getitem(new_idx)
    
    def getitem(self, idx):

        file = self.samples[idx]
        sample = json.load(open(os.path.join(self.root, file), 'r'))

        # # debug
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=((len(sample['pseudo_label']) + len(sample['human_label']))*6, 6))
        # for i, pseudo_label in enumerate(sample['pseudo_label']):
        #     plt.subplot(1, len(sample['pseudo_label']) + len(sample['human_label']), i+1)
        #     plt.imshow(decode(pseudo_label['label']), cmap='Reds')
        #     plt.title(pseudo_label['source'])

        # for i, human_label in enumerate(sample['human_label']):
        #     plt.subplot(1, len(sample['pseudo_label']) + len(sample['human_label']), i+1+len(sample['pseudo_label']))
        #     plt.imshow(decode(human_label['label']), cmap='Greens')
        #     plt.title(human_label['source'])
        # plt.show()

        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution))

        if self.do_augmentation:
            image = augmentation(image)

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