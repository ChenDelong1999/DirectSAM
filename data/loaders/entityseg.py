import os
import json
from PIL import Image
import numpy as np
from pycocotools.mask import decode

def decode_and_merge_rle_annotations(annotations):
    masks = decode(annotations[0])
    for i in range(1, len(annotations)):
        masks += decode(annotations[i])*(i+1)
    return masks


class EntitySegDataset():

    def __init__(self, root, split, lr=True):

        split = 'val' if split == 'validation' else 'train'

        if lr:
            annotation_file = os.path.join(root, f'entityseg_{split}_lr.json')
            image_folder = os.path.join(root, 'images_lr')
        else:
            annotation_file = os.path.join(root, f'entityseg_{split}.json')
            image_folder = os.path.join(root, 'images')

        annotation_file = json.load(open(annotation_file, 'r'))
        
        images = {}
        labels = {}
        for image in annotation_file['images']:
            images[image['id']] = os.path.join(image_folder, image['file_name'])
            labels[image['id']] = []

        for label in annotation_file['annotations']:
            labels[label['image_id']].append(label['segmentation'])

        self.image_paths = list(images.values())
        self.labels = [labels[image_id] for image_id in images.keys()]


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx]).convert("RGB")
        labels = self.labels[idx]
        label_map = decode_and_merge_rle_annotations(labels)

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed

    def set_transform(self, transform):
        self.transform = transform
        