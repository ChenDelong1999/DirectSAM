
from lvis import LVIS
from PIL import Image as PILImage
import os



class LvisDataset:
    def __init__(self, image_folder, annotation_file):

        self.lvis = LVIS(annotation_file)
        self.image_dir = image_folder
        self.image_ids = self.lvis.get_img_ids()

        self.image_paths = []
        for img_id in self.image_ids:
            img_info = self.lvis.load_imgs([img_id])[0]
            img_path = os.path.join(self.image_dir, img_info['coco_url'].replace('http://images.cocodataset.org/', ''))
            self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        img_info = self.lvis.load_imgs([img_id])[0]
        img_path = os.path.join(self.image_dir, img_info['coco_url'].replace('http://images.cocodataset.org/', ''))
        image = PILImage.open(img_path).convert("RGB")

        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        anns = self.lvis.load_anns(ann_ids)

        label_map = None
        for i, ann in enumerate(anns):
            mask = self.lvis.ann_to_mask(ann)
            if label_map is None:
                label_map = mask
            else:
                label_map += mask * (i + 1)

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed

    def set_transform(self, transform):
        self.transform = transform