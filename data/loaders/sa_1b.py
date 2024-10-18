
import os
import tqdm
from PIL import Image
import json
from pycocotools.mask import decode

class SA1BDataset:

    def __init__(self, root):
        if os.path.exists('sa1b_image_paths.json'):
            self.image_paths = json.load(open('sa1b_image_paths.json'))
        else:
            self.image_paths = []
            for subfolder in tqdm.tqdm(os.listdir(root)):
                subfolder_path = os.path.join(root, subfolder)
                if os.path.isdir(subfolder_path):
                    for img_file in os.listdir(subfolder_path):

                        if img_file.endswith('.jpg'):
                            image_path = os.path.join(subfolder_path, img_file)
                            json_path = image_path.replace('.jpg', '.json')
                            if os.path.exists(json_path):
                                self.image_paths.append(image_path)
            json.dump(self.image_paths, open('sa1b_image_paths.json', 'w'))


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")

            json_path = self.image_paths[idx].replace('.jpg', '.json')
            annotations = json.load(open(json_path))['annotations']

            label_map = None
            for i, annotation in enumerate(annotations):
                segmentation = decode(annotation['segmentation'])
                if label_map is None:
                    label_map = segmentation
                else:
                    label_map += segmentation * ((i + 1) % 255)

            batch = {'image': [image], 'annotation': [label_map]}
            transformed = self.transform(batch)
            for key in transformed:
                transformed[key] = transformed[key][0]

            return transformed
        
        except Exception as e:
            print(f"Error with {idx}th image {self.image_paths[idx]}:\n{e}")
            new_idx = idx + 1 if idx + 1 < len(self) else 0
            return self.__getitem__(new_idx)


    def set_transform(self, transform):
        self.transform = transform