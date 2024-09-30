import os
from PIL import Image as PILImage


class PascalPanopticPartsDataset():
    def __init__(self, image_folder, label_folder, split):

        if split == 'train':
            self.annotation_path = os.path.join(label_folder, 'training')
        else:
            self.annotation_path = os.path.join(label_folder, 'validation')

        self.samples = os.listdir(self.annotation_path)
        self.image_folder = image_folder

        self.image_paths = [os.path.join(self.image_folder, sample.split('.')[0]+'.jpg') for sample in self.samples]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        id = sample.split('.')[0]

        image = PILImage.open(os.path.join(self.image_folder, id+'.jpg')).convert('RGB')
        label_map = PILImage.open(os.path.join(self.annotation_path, sample))

        batch = {'image': [image], 'annotation': [label_map]}
        transformed = self.transform(batch)
        for key in transformed:
            transformed[key] = transformed[key][0]

        return transformed

    def set_transform(self, transform):
        self.transform = transform
        