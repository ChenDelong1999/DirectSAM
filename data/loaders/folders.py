import os
import tqdm
from PIL import Image
import numpy as np
import json


class FoldersDataset:

    def __init__(
            self, 
            image_folder, 
            label_folder, 
            image_suffix='jpg', 
            label_suffix='png',
            **kwargs
            ):
        
        image_files = [x for x in os.listdir(image_folder) if x.split('.')[-1] == image_suffix]
        image_files = sorted(image_files)

        self.image_paths, self.label_paths = [], []

        for image_file in tqdm.tqdm(image_files):
            if not image_file.endswith(image_suffix):
                continue

            id = image_file.split('.')[0]
            label_file = id + '.' + label_suffix
            if not os.path.exists(os.path.join(label_folder, label_file)):
                continue

            self.image_paths.append(os.path.join(image_folder, image_file))
            self.label_paths.append(os.path.join(label_folder, label_file))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label_map = Image.open(self.label_paths[idx]) # don't do convert("L") here

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed


    def set_transform(self, transform):
        self.transform = transform
        

class CIHPDataset(FoldersDataset):

    def __init__(self, root, split):
        split = 'Training' if split == 'train' else 'Validation'
        image_folder = os.path.join(root, split, 'Images')
        label_folder = os.path.join(root, split, 'Instance_ids')

        super().__init__(image_folder, label_folder)

class EgoHOSDataset(FoldersDataset):

    def __init__(self, root, split):
        split = 'val' if split == 'validation' else 'train'
        image_folder = os.path.join(root, split, 'image')
        label_folder = os.path.join(root, split, 'label')

        super().__init__(image_folder, label_folder)

class PhenoBenchDataset(FoldersDataset):

    def __init__(self, root, split):
        split = 'val' if split == 'validation' else 'train'
        image_folder = os.path.join(root, split, 'images')
        label_folder = os.path.join(root, split, 'leaf_instances')

        super().__init__(image_folder, label_folder, image_suffix='png')


class SUIMDataset(FoldersDataset):

    def __init__(self, root, split):
        image_folder = os.path.join(root, split, 'images')
        label_folder = os.path.join(root, split, 'masks')

        super().__init__(image_folder, label_folder)


class MyFoodDataset(FoldersDataset):

    def __init__(self, root, split):
        split = 'val' if split == 'validation' else 'train'
        image_folder = os.path.join(root, split)
        label_folder = os.path.join(root, split+'_ann')

        super().__init__(image_folder, label_folder, image_suffix='png')


class ISAIDDataset(FoldersDataset):

    def __init__(self, root, split):
        split = 'val' if split == 'validation' else 'train'
        image_folder = os.path.join(root, split, 'images')
        label_folder = os.path.join(root, split, 'labels')

        super().__init__(image_folder, label_folder)


class MapillaryMetropolisDataset(FoldersDataset):

    def __init__(self, root, split):
        split = 'val' if split == 'validation' else 'train'
        image_folder = os.path.join(root, split, 'images')
        label_folder = os.path.join(root, split, 'labels')

        super().__init__(image_folder, label_folder)


class LIPDataset(FoldersDataset):

    def __init__(self, root, split):
        if split == 'validation':
            image_folder = os.path.join(root, 'val_images')
            label_folder = os.path.join(root, 'TrainVal_parsing_annotations/val_segmentations')
        else:
            image_folder = os.path.join(root, 'train_images')
            label_folder = os.path.join(root, 'TrainVal_parsing_annotations/train_segmentations')
            
        super().__init__(image_folder, label_folder)



class CocoNutLDataset(FoldersDataset):
    
    def __init__(
        self, 
        image_folder, 
        label_folder,
        ):
    
        id_to_path = {}
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.endswith(".jpg"):
                    id_to_path[file.split('.')[0]] = os.path.join(root, file)

        labels = os.listdir(label_folder)

        self.label_paths = []
        self.image_paths = []

        for label in tqdm.tqdm(labels):
            label_path = os.path.join(label_folder, label)
            if not os.path.exists(label_path):
                continue
            self.label_paths.append(label_path)
            id = label.split(".")[0]
            self.image_paths.append(id_to_path[id])


class LoveDADataset(FoldersDataset):

    def __init__(self, root, split):

        if split == 'validation':
            split = 'Val'
        elif split == 'train':
            split = 'Train'

        self.image_paths = []
        self.label_paths = []
        for domain in ['Urban', 'Rural']:
            images = os.listdir(os.path.join(root, split, domain, 'images_png'))
            labels = os.listdir(os.path.join(root, split, domain, 'masks_png'))
            self.image_paths += [os.path.join(root, split, domain, 'images_png', x) for x in images]
            self.label_paths += [os.path.join(root, split, domain, 'masks_png', x) for x in labels]

class SPINDataset(FoldersDataset):

    def __init__(self, image_folder, label_folder, split):

        if split == 'train':
            label_folder = os.path.join(label_folder, 'train')
        elif split == 'validation':
            label_folder = os.path.join(label_folder, 'test') # 'val' split ignored

        self.samples = []
        for sample in os.listdir(label_folder):
            self.samples.append(sample.split('.')[0])

        self.image_paths = [os.path.join(image_folder, id.split('_')[0], id+'.JPEG') for id in self.samples]
        self.label_paths = [os.path.join(label_folder, id+'.png') for id in self.samples]

class UAVIDDataset(FoldersDataset):

    def __init__(self, root, split):

        split = 'val' if split == 'validation' else 'train'

        sequences = os.listdir(os.path.join(root, f'uavid_{split}'))

        self.image_paths = []
        self.label_paths = []

        for sequence in sequences:
            images = os.listdir(os.path.join(root, f'uavid_{split}', sequence, 'Images'))
            
            for image in images:
                self.image_paths.append(os.path.join(root, f'uavid_{split}', sequence, 'Images', image))
                self.label_paths.append(os.path.join(root, f'uavid_{split}', sequence, 'Labels', image))



class SOBADataset(FoldersDataset):

    def __init__(self, root, split):

        self.root = root
        split_ratio = 0.84

        self.samples = []
        for subset_name in os.listdir(root):
            subset_samples = []
            for file in os.listdir(os.path.join(root, subset_name)):
                if file.endswith('.jpg'):
                    subset_samples.append(os.path.join(subset_name, file.split('.')[0]))

            split_index = int(len(subset_samples) * split_ratio)
            if split == 'train':
                self.samples += subset_samples[:split_index]
            else:
                self.samples += subset_samples[split_index:]

        self.image_paths = [os.path.join(root, sample+'.jpg') for sample in self.samples]

    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        sample = self.samples[idx]
        label_instance = np.array(Image.open(os.path.join(self.root, sample+'-2.png')).convert("L"))
        label_shadow = np.array(Image.open(os.path.join(self.root, sample+'-3.png')).convert("L"))

        label_shadow[label_shadow>0] += np.max(label_instance) + 1
        label_map = np.zeros_like(label_instance)
        label_map[label_instance>0] = label_instance[label_instance>0]
        label_map[label_shadow>0] = label_shadow[label_shadow>0]

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed


    def set_transform(self, transform):
        self.transform = transform
