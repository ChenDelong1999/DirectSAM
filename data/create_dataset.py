
import os
from datasets import load_dataset

from .loaders import *
from .transforms import transforms_for_labelmap_dataset


def create_dataset(dataset_info, split, resolution, thickness=3):

    if dataset_info['type'] == 'DSA':
        return DSADataset(**dataset_info, resolution=resolution)
    
    # belowings are all label map-based datasets

    assert split in ['train', 'validation']

    label_map_mode = 'single_channel'

    if dataset_info['type'] == 'SA1B':
        dataset = SA1BDataset(dataset_info['root'])

    elif dataset_info['type'] == 'folders':

        dataset = FoldersDataset(**dataset_info)

        for keyword in ['COIFT', 'DIS5K-DIS-TR', 'DIS5K-DIS-VD', 'DUTS-TE', 'DUTS-TR', 'ecssd', 'fss_all', 'HRSOD', 'MSRA_10K', 'ThinObject5K']:
            if keyword in dataset_info['image_folder']:
                label_map_mode = 'force_binary'

        if 'GTA' in dataset_info['image_folder'] and split == 'validation':
                print('GTA-V dataset only supports train split')

    elif dataset_info['type'] == 'huggingface_dataset':
        dataset = load_dataset(dataset_info['id'], split=split) 
        
        if os.path.exists(dataset_info['image_folder']):
            images = os.listdir(dataset_info['image_folder'])
            dataset.image_paths = [os.path.join(dataset_info['image_folder'], x) for x in images]
        else:
            dataset.image_paths = []
    
    elif dataset_info['type'] == "COCONut-l":
        if split == 'validation':
            print('COCONut-l dataset only supports train split')

        dataset = CocoNutLDataset(dataset_info['image_folder'], dataset_info['label_folder'])

    elif dataset_info['type'] == 'PascalPanopticParts':
        dataset = PascalPanopticPartsDataset(dataset_info['image_folder'], dataset_info['label_folder'], split)
        
    elif dataset_info['type'] == 'CelebA':
        if split == 'validation':
            print('CelebA dataset only supports train split')
        dataset = CelebADataset(root=dataset_info['root'])

    elif dataset_info['type'] == 'EgoHOS':
        dataset = EgoHOSDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'PhenoBench':
        dataset = PhenoBenchDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'UAVID':
        dataset = UAVIDDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'EntitySeg':
        dataset = EntitySegDataset(dataset_info['root'], split, dataset_info['lr'])

    elif dataset_info['type'] == 'CIHP':
        dataset = CIHPDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'MyFood':
        dataset = MyFoodDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'LIP':
        dataset = LIPDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'SOBA':
        dataset = SOBADataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'SUIM':
        dataset = SUIMDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'LoveDA':
        dataset = LoveDADataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'SPIN':
        dataset = SPINDataset(dataset_info['image_folder'], dataset_info['label_folder'], split)

    elif dataset_info['type'] == 'Fashionpedia':
        dataset = FashionpediaDataset(dataset_info['root'], split)
    
    elif dataset_info['type'] == 'PartImageNetPP':
        dataset = PartImageNetPPDataset(dataset_info['image_folder'], dataset_info['label_folder'], split)

    elif dataset_info['type'] == 'SeginW':
        dataset = SeginWDataset(dataset_info['root'], split)

    elif dataset_info['type'] == 'LVIS':
        annotation_file = 'lvis_v1_val.json' if split == 'validation' else 'lvis_v1_train.json'
        dataset = LvisDataset(
            dataset_info['image_folder'], 
            os.path.join(dataset_info['label_folder'], annotation_file)
            )
    
    elif dataset_info['type'] == 'PACO':
        annotation_file = 'paco_lvis_v1_val.json' if split == 'validation' else 'paco_lvis_v1_train.json'
        dataset = LvisDataset(
            dataset_info['image_folder'], 
            os.path.join(dataset_info['label_folder'], annotation_file)
            )
        
    else:
        print(f"Unknown dataset type: {dataset_info['type']}")
    
    dataset.set_transform(
        lambda x: transforms_for_labelmap_dataset(
            batch=x, resolution=resolution, thickness=thickness, label_map_mode=label_map_mode,
            **dataset_info
            ))

    return dataset