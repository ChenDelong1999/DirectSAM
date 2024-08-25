
import os
import json
from PIL import Image as PILImage
import numpy as np
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
    label_map_to_boundary
    )

import os
import json
from datasets import load_dataset, Dataset
from torchvision.datasets import CocoDetection


def create_dataset(dataset_info, split, resolution, thickness=3):

    assert split in ['train', 'validation']

    if dataset_info['type'] == 'CIHP':
        split = 'Training' if split == 'train' else 'Validation'
        image_folder = os.path.join(dataset_info['root'], split, 'Images')
        label_folder = os.path.join(dataset_info['root'], split, 'Instance_ids')

        return get_image_folder_dataset(image_folder, label_folder, resolution, thickness, image_suffix='jpg', label_suffix='png')
    
    elif dataset_info['type'] == 'SeginW':
        return SeginW(
            root=dataset_info['root'],
            resolution=resolution,
            thickness=thickness,
            split=split
        )
    
    elif dataset_info['type'] == 'UDA-Part':

        if split!='train':
            print('Warning: UDA-Part dataset only supports train split')
        
        return UDAPart(
            root=dataset_info['root'],
            resolution=resolution,
            thickness=thickness
        )
    elif dataset_info['type'] == 'Fashionpedia':
        return Fashionpedia(
            root=dataset_info['root'],
            resolution=resolution,
            thickness=thickness,
            split=split
        )
    

    elif dataset_info['type'] == 'PartIT':
            
        return PartIT(
            root=dataset_info['root'],
            resolution=resolution,
            thickness=thickness,
            split=split
        )
    
    elif dataset_info['type'] == 'SPIN':
            
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
        dataset = get_image_folder_dataset(dataset_info['image_folder'], dataset_info['label_folder'], resolution, thickness)

        return dataset
    
    else:
        raise ValueError(f"Dataset type {dataset_info['type']} not supported")
    

def get_image_folder_dataset(image_folder,label_folder, resolution, thickness, image_suffix='jpg', label_suffix='png'):
    image_files = [x.split('.')[0] for x in os.listdir(image_folder) if x.split('.')[-1] == image_suffix]
    image_ids = [file_name.split('.')[0] for file_name in image_files]

    # label_files = [x.split('.')[0] for x in os.listdir(label_folder) if x.split('.')[-1] == label_suffix]
    # label_ids = [file_name.split('.')[0] for file_name in label_files]

    image_ids_with_path = [os.path.join(image_folder, x) for x in image_ids]
    label_ids_with_path = [os.path.join(label_folder, x) for x in image_ids]

    dataset = Dataset.from_dict({"image": sorted(image_ids_with_path), "label": sorted(label_ids_with_path)})
    dataset.set_transform(lambda x: transforms_image_folders(x, resolution, thickness, image_suffix, label_suffix))

    return dataset


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

        boundary = label_map_to_boundary(label_map, self.resolution, self.thickness)
        return {'image': image, 'label': boundary}


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
        
        label_map = coco_annotation_to_label_map(anns, coco)
        boundary = label_map_to_boundary(label_map, self.resolution, self.thickness)
        return {'image': image, 'label': boundary}
    


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
        label = label_map_to_boundary(label, self.resolution, self.thickness)

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
        label = label_map_to_boundary(label, self.resolution, self.thickness)

        return {'image': image, 'label': label}
    

class PartIT(TorchDataset):

    def __init__(self, root, resolution, thickness, split):

        self.root = root
        self.resolution = resolution
        self.thickness = thickness
        self.split = 'test' if split == 'validation' else 'train'

        self.samples = []
        for class_folder in os.listdir(root):
            for sample in os.listdir(os.path.join(root, class_folder, self.split)):
                self.samples.append(os.path.join(class_folder, self.split, sample))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        sample = self.samples[idx]
        image = PILImage.open(os.path.join(self.root, sample, '0.png')).convert('RGB')
        image = resize_image(image, self.resolution)

        files = os.listdir(os.path.join(self.root, sample))
        # get all images where has 'occluded.png

        label_map = None
        part_id = 1
        for file in files:
            if 'occluded.png' in file:
                label = PILImage.open(os.path.join(self.root, sample, file))
                label = (np.array(label)!= 255) * part_id

                if label_map is None:
                    label_map = label
                else:
                    label_map += label
                part_id += 1
        
        label = label_map_to_boundary(label_map, self.resolution, self.thickness)

        return {'image': image, 'label': label}

class Fashionpedia(TorchDataset):

    def __init__(self, root, resolution, thickness, split):

        self.root = root
        self.resolution = resolution
        self.thickness = thickness
        self.split = 'test' if split == 'validation' else 'train'

        self.dataset = COCO(os.path.join(root, f"instances_attributes_{'val' if split == 'validation' else 'train'}2020.json"))

        self.img_ids = self.dataset.getImgIds()
        self.cat_ids = self.dataset.getCatIds()

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        image = PILImage.open(os.path.join(self.root, self.split, self.dataset.loadImgs(img_id)[0]['file_name'])).convert('RGB')

        image = resize_image(image, self.resolution)

        ann_ids = self.dataset.getAnnIds(imgIds=img_id)
        anns = self.dataset.loadAnns(ann_ids)

        label_map = coco_annotation_to_label_map(anns, self.dataset)
        boundary = label_map_to_boundary(label_map, self.resolution, self.thickness)

        return {'image': image, 'label': boundary}


def coco_annotation_to_label_map(anns, coco):
    label_map = None
    for i, ann in enumerate(anns):
        mask = coco.annToMask(ann)
        if label_map is None:
            label_map = mask
        else:
            label_map += mask * (i + 1)

    return label_map


class UDAPart(TorchDataset):

    def __init__(self, root, resolution, thickness):

        self.resolution = resolution
        self.thickness = thickness

        self.samples = []
        for class_folder in os.listdir(root):
            image_folder = os.path.join(root, class_folder, 'image')
            for model in os.listdir(image_folder):
                for image in os.listdir(os.path.join(image_folder, model)):
                    self.samples.append(os.path.join(image_folder, model, image))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
            
        image_path = self.samples[idx]
        image = PILImage.open(image_path).convert('RGB')
        image = resize_image(image, self.resolution)

        label_path = image_path.replace('/image/', '/seg/')
        label = PILImage.open(label_path)
        label = label_map_to_boundary(label, self.resolution, self.thickness)

        return {'image': image, 'label': label}
    

class SeginW():

    def __init__(self, root, resolution, thickness, split):

        self.resolution = resolution
        self.thickness = thickness
        self.split = 'valid' if split == 'validation' else 'train'
        self.root = root

        self.subsets = {}
        for subset_name in os.listdir(root):
            if subset_name == 'Salmon-Fillet':
                continue # ignore this subset due to quality issues
            for file in os.listdir(os.path.join(root, subset_name, self.split)):
                if file.endswith('.json'):
                    self.subsets[subset_name] = COCO(os.path.join(root, subset_name, self.split, file))


        self.samples = []
        for subset_name, subset in self.subsets.items():
            for img_id in subset.getImgIds():
                self.samples.append((subset_name, img_id))

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):

        subset_name, img_id = self.samples[idx]
        img = self.subsets[subset_name].loadImgs(img_id)[0]
        ann_ids = self.subsets[subset_name].getAnnIds(imgIds=img['id'])
        anns = self.subsets[subset_name].loadAnns(ann_ids)

        img_path = os.path.join(self.root, subset_name, self.split, img['file_name'])
        image = PILImage.open(img_path).convert('RGB')
        image = resize_image(image, self.resolution)
        
        if len(anns) > 0:
            label_map = coco_annotation_to_label_map(anns, self.subsets[subset_name])
        else:
            label_map = np.zeros((img['height'], img['width']), dtype=np.uint8)
            print(f"Warning: Empty annotation for {img_path}")

        boundary = label_map_to_boundary(label_map, self.resolution, self.thickness)

        return {'image': image, 'label': boundary}
        