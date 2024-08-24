
import os
import json
from PIL import Image as PILImage
from datasets import Dataset, load_dataset
from torchvision.datasets import CocoDetection

from lvis import LVIS
from pycocotools.coco import COCO
from torch.utils.data import Dataset as TorchDataset

from .transforms import (
    transforms_huggingface_dataset, 
    transforms_image_folders, 
    transforms_coco_single_sample, 
    transforms_entity_seg, 
    transforms_directsam_pseudo_label,
    resize_image,
    annotation_to_label
    )

import os
import json
from datasets import load_dataset, Dataset
from torchvision.datasets import CocoDetection


def create_dataset(dataset_info, split, resolution, thickness=3):

    assert split in ['train', 'validation']

    if dataset_info['type'] == 'SPIN':
            
        return SPIN(
            annotation_path=dataset_info['annotations'],
            image_dir=dataset_info['image_folder'],
            resolution=resolution,
            thickness=thickness,
            split=split
        )
    
    elif dataset_info['type'] == 'LVIS':
        annotation_file = 'lvis_v1_val.json' if split == 'validation' else 'lvis_v1_train.json'
            
        return LvisDataset(
            annotation_path=os.path.join(dataset_info['annotations'], annotation_file),
            image_dir=dataset_info['image_folder'],
            resolution=resolution,
            thickness=thickness
        )
    
    elif dataset_info['type'] == 'PascalPanopticParts':

        return PascalPanopticParts(
            annotation_path=dataset_info['annotations'],
            image_dir=dataset_info['image_folder'],
            resolution=resolution,
            thickness=thickness,
            split=split
        )

    elif dataset_info['type'] == 'PartImageNet++':
                    
        return PartImageNetPP(
            annotation_path=dataset_info['annotations'],
            image_dir=dataset_info['image_folder'],
            resolution=resolution,
            thickness=thickness,
            split=split
        )

    elif dataset_info['type'] == 'directsam_pseudo_label':

        image_paths = [] 
        json_files = []
        for root, dirs, files in os.walk(dataset_info['pseudo_label_dir']):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    json_files.append(json_file)
                    image_paths.append(json_file.replace(dataset_info['pseudo_label_dir'], dataset_info['sa1b_dir']).replace(".json", ".jpg"))

        print(f"Found {len(json_files)} json files")

        dataset = Dataset.from_dict({"image": image_paths, "label": json_files})
        dataset.set_transform(lambda x: transforms_directsam_pseudo_label(x, resolution, thickness))

        return dataset
    
    elif dataset_info['type'] == 'huggingface_dataset': # ade_20k
        
        dataset = load_dataset(dataset_info['id'], split=split) # ['train', 'test', 'validation']
        dataset.set_transform(lambda x: transforms_huggingface_dataset(x, resolution, thickness))
        return dataset

    elif dataset_info['type'] == 'EntitySeg':

        split = 'val' if split == 'validation' else 'train'

        if dataset_info['lr']:
            annotation_file = os.path.join(dataset_info['root'], f'entityseg_{split}_lr.json')
            image_folder = os.path.join(dataset_info['root'], 'images_lr')
        else:
            annotation_file = os.path.join(dataset_info['root'], f'entityseg_{split}.json')
            image_folder = os.path.join(dataset_info['root'], 'images')

        annotation_file = json.load(open(annotation_file, 'r'))
        
        images = {}
        labels = {}
        for image in annotation_file['images']:
            images[image['id']] = os.path.join(image_folder, image['file_name'])
            labels[image['id']] = []

        for label in annotation_file['annotations']:
            labels[label['image_id']].append(label['segmentation'])

        images_list = list(images.values())
        labels_list = [labels[image_id] for image_id in images.keys()]

        dataset = Dataset.from_dict({"image": images_list, "label": labels_list})
        dataset.set_transform(lambda x: transforms_entity_seg(x, resolution, thickness))

        return dataset
    
    elif dataset_info['type'] == 'coco':
        if split=='validation':
            dataset =  CocoDetection(
                os.path.join(dataset_info['image_folder'], 'val2017'), 
                os.path.join(dataset_info['annotations'], 'instances_val2017.json'),
                transforms=lambda img, ann: transforms_coco_single_sample(img, ann, resolution, thickness)
                )
        elif split=='train':
            dataset =  CocoDetection(
                os.path.join(dataset_info['image_folder'], 'train2017'), 
                os.path.join(dataset_info['annotations'], 'instances_train2017.json'),
                transforms=lambda img, ann: transforms_coco_single_sample(img, ann, resolution, thickness)
                )

        return dataset
    
    elif dataset_info['type'] == 'LoveDA':

        if split == 'validation':
            split = 'Val'
        elif split == 'train':
            split = 'Train'

        image_paths = []
        label_paths = []
        for domain in ['Urban', 'Rural']:
            images = os.listdir(os.path.join(dataset_info['root'], split, domain, 'images_png'))
            labels = os.listdir(os.path.join(dataset_info['root'], split, domain, 'masks_png'))
            image_paths += [os.path.join(dataset_info['root'], split, domain, 'images_png', x) for x in images]
            label_paths += [os.path.join(dataset_info['root'], split, domain, 'masks_png', x) for x in labels]

        dataset = Dataset.from_dict({"image": image_paths, "label": label_paths})
        dataset.set_transform(lambda x: transforms_image_folders(x, resolution, thickness, image_suffix='', label_suffix=''))

        return dataset
    
    elif dataset_info['type'] == 'image_folders':

        print('Warning: "split" argument is ignored for image_folders datasets')

        label_files = [x.split('.')[0] for x in os.listdir(dataset_info['label_folder']) if x.split('.')[-1] == 'png']
        label_ids = [file_name.split('.')[0] for file_name in label_files]

        image_files = [x.split('.')[0] for x in os.listdir(dataset_info['image_folder']) if x.split('.')[-1] == 'jpg']
        image_ids = [file_name.split('.')[0] for file_name in image_files]

        # use image_ids to filter out label_ids, since for COCONut-B dataset, the label folder also contains samples in COCONut-B
        image_ids_with_path = [os.path.join(dataset_info['image_folder'], x) for x in image_ids]
        label_ids_with_path = [os.path.join(dataset_info['label_folder'], x) for x in image_ids]

        dataset = Dataset.from_dict({"image": sorted(image_ids_with_path), "label": sorted(label_ids_with_path)})
        dataset.set_transform(lambda x: transforms_image_folders(x, resolution, thickness))

        return dataset
    
    else:
        raise ValueError(f"Dataset type {dataset_info['type']} not supported")
    



class LvisDataset(TorchDataset):
    def __init__(self, annotation_path, image_dir, resolution, thickness):
        self.lvis = LVIS(annotation_path)
        self.image_dir = image_dir
        self.image_ids = self.lvis.get_img_ids()
        self.resolution = resolution
        self.thickness = thickness

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        img_info = self.lvis.load_imgs([img_id])[0]
        img_path = os.path.join(self.image_dir, img_info['coco_url'].replace('http://images.cocodataset.org/', ''))
        image = PILImage.open(img_path).convert("RGB")
        image = resize_image(image, self.resolution)

        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        anns = self.lvis.load_anns(ann_ids)

        label_map = None
        for i, ann in enumerate(anns):
            mask = self.lvis.ann_to_mask(ann)
            if label_map is None:
                label_map = mask
            else:
                label_map += mask * (i + 1)

        label_map = annotation_to_label(label_map, self.resolution, self.thickness)
        return {'image': image, 'label': label_map}


class PartImageNetPP(TorchDataset):
    def __init__(self, annotation_path, image_dir, resolution, thickness, split):

        self.subsets = []
        for json_path in os.listdir(annotation_path):
            self.subsets.append(COCO(os.path.join(annotation_path, json_path)))

        self.image_dir = image_dir
        self.resolution = resolution
        self.thickness = thickness

        self.split = split
        if split == 'train':
            self.samples_per_class = 90
        else:
            self.samples_per_class = 10

    def __len__(self):
        return self.samples_per_class * len(self.subsets)

    def __getitem__(self, idx):
        subset_index = idx // self.samples_per_class
        img_id = idx % self.samples_per_class

        img_id += 90 if self.split == 'validation' else 0

        coco = self.subsets[subset_index]
        catIds = coco.getCatIds(catNms=['str']) 
        imgIds = coco.getImgIds(catIds=catIds ) 

        img = coco.loadImgs(imgIds[img_id-1])[0]  
        image_name = img['file_name']
        image = PILImage.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        image = resize_image(image, self.resolution)

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        label_map = None
        for i, ann in enumerate(anns):
            mask = coco.annToMask(ann)
            if label_map is None:
                label_map = mask
            else:
                label_map += mask * (i + 1)

        label_map = annotation_to_label(label_map, self.resolution, self.thickness)
        return {'image': image, 'label': label_map}
    


class PascalPanopticParts(TorchDataset):
    def __init__(self, annotation_path, image_dir, resolution, thickness, split):

        if split == 'train':
            self.annotation_path = os.path.join(annotation_path, 'training')
        else:
            self.annotation_path = os.path.join(annotation_path, 'validation')

        self.samples = os.listdir(self.annotation_path)
        
        self.image_dir = image_dir
        self.resolution = resolution
        self.thickness = thickness

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        id = sample.split('.')[0]

        image = PILImage.open(os.path.join(self.image_dir, id+'.jpg')).convert('RGB')
        image = resize_image(image, self.resolution)

        label = PILImage.open(os.path.join(self.annotation_path, sample))
        label = annotation_to_label(label, self.resolution, self.thickness)

        return {'image': image, 'label': label}

class SPIN(TorchDataset):

    def __init__(self, annotation_path, image_dir, resolution, thickness, split):

        if split == 'train':
            self.annotation_path = os.path.join(annotation_path, 'train')
        elif split == 'validation':
            self.annotation_path = os.path.join(annotation_path, 'test') # 'val' split ignored

        self.image_dir = image_dir
        self.resolution = resolution
        self.thickness = thickness
        self.split = split

        self.samples = []
        for sample in os.listdir(self.annotation_path):
            self.samples.append(sample.split('.')[0])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        id = self.samples[idx]

        image = PILImage.open(os.path.join(self.image_dir, id.split('_')[0], id+'.JPEG')).convert('RGB')
        image = resize_image(image, self.resolution)

        label = PILImage.open(os.path.join(self.annotation_path, id+'.png'))
        label = annotation_to_label(label, self.resolution, self.thickness)

        return {'image': image, 'label': label}