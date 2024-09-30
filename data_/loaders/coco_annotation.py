
import os
from pycocotools.coco import COCO
from PIL import Image as PILImage
import numpy as np

def coco_annotation_to_label_map(anns, coco):
    label_map = None
    for i, ann in enumerate(anns):
        mask = coco.annToMask(ann)
        if label_map is None:
            label_map = mask
        else:
            label_map += mask * (i + 1)

    return label_map



class FashionpediaDataset:

    def __init__(self, root, split):

        self.root = root
        self.split = 'test' if split == 'validation' else 'train'

        self.dataset = COCO(os.path.join(root, f"instances_attributes_{'val' if split == 'validation' else 'train'}2020.json"))

        self.img_ids = self.dataset.getImgIds()
        self.cat_ids = self.dataset.getCatIds()

        self.image_paths = [os.path.join(root, self.split, self.dataset.loadImgs(img_id)[0]['file_name']) for img_id in self.img_ids]

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        image = PILImage.open(os.path.join(self.root, self.split, self.dataset.loadImgs(img_id)[0]['file_name'])).convert('RGB')

        ann_ids = self.dataset.getAnnIds(imgIds=img_id)
        anns = self.dataset.loadAnns(ann_ids)

        label_map = coco_annotation_to_label_map(anns, self.dataset)

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed

    def set_transform(self, transform):
        self.transform = transform
        


class PartImageNetPPDataset:
    def __init__(self, image_folder, label_folder, split):

        self.subsets = []
        for json_path in os.listdir(label_folder):
            self.subsets.append(COCO(os.path.join(label_folder, json_path)))

        self.image_folder = image_folder

        self.split = split
        if split == 'train':
            self.samples_per_class = 90
        else:
            self.samples_per_class = 10

        self.image_paths = self.prepare_image_paths()

    def prepare_image_paths(self):
        image_paths = []
        for idx in range(len(self)):
            subset_index = idx // self.samples_per_class
            img_id = idx % self.samples_per_class

            img_id += 90 if self.split == 'validation' else 0

            coco = self.subsets[subset_index]
            catIds = coco.getCatIds(catNms=['str'])
            imgIds = coco.getImgIds(catIds=catIds)

            img = coco.loadImgs(imgIds[img_id - 1])[0]
            image_name = img['file_name']

            image_path = os.path.join(self.image_folder, image_name)
            image_paths.append(image_path)

        return image_paths

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
        image = PILImage.open(os.path.join(self.image_folder, image_name)).convert('RGB')

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        label_map = coco_annotation_to_label_map(anns, coco)
        
        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed

    def set_transform(self, transform):
        self.transform = transform
        

    

class SeginWDataset:

    def __init__(self, root, split):

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

        self.image_paths = [os.path.join(root, subset_name, self.split, self.subsets[subset_name].loadImgs(img_id)[0]['file_name']) for subset_name, img_id in self.samples]

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):

        subset_name, img_id = self.samples[idx]
        img = self.subsets[subset_name].loadImgs(img_id)[0]
        ann_ids = self.subsets[subset_name].getAnnIds(imgIds=img['id'])
        anns = self.subsets[subset_name].loadAnns(ann_ids)

        img_path = os.path.join(self.root, subset_name, self.split, img['file_name'])
        image = PILImage.open(img_path).convert('RGB')
        
        if len(anns) > 0:
            label_map = coco_annotation_to_label_map(anns, self.subsets[subset_name])
        else:
            label_map = np.zeros((img['height'], img['width']), dtype=np.uint8)
            print(f"Warning: Empty annotation for {img_path}")

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed

    def set_transform(self, transform):
        self.transform = transform