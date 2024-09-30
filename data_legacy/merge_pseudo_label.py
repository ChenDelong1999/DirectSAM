import os
import numpy as np
import json
from pycocotools.mask import encode, decode
from transforms import label_map_to_boundary
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_sa1b_annotations(annotations, resolution, thickness):
    h, w = annotations[0]['segmentation']['size']
    label_map = np.zeros((h, w))
    id = 0
    for annotation in annotations:
        segmentation = decode(annotation['segmentation'])
        label_map[segmentation == 1] = id
        id += 1
    boundary = label_map_to_boundary(label_map, resolution=resolution, thickness=thickness)
    return boundary, h, w

def process_file(args):
    json_file, image_path, sa1b_anns, resolution, thickness, pseudo_dir, output_dir = args

    results = json.load(open(json_file, "r"))
    boundary = decode(results)
    output_file = json_file.replace(pseudo_dir, output_dir)

    if os.path.exists(output_file):
        return
    try:

        annotations = json.load(open(sa1b_anns, "r"))['annotations']
        annotations, h, w = load_sa1b_annotations(annotations, resolution=resolution, thickness=thickness)

        merged = (boundary + annotations) > 0
        merged[:thickness] = True
        merged[-thickness:] = True
        merged[:, :thickness] = True
        merged[:, -thickness:] = True

        # resize merged to h, w
        merged = cv2.resize(merged.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        rle = encode(np.array(merged, order='F', dtype=np.uint8))
        rle['counts'] = rle['counts'].decode('utf-8')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(rle, f, indent=4)
    
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

def main():
    sa1b_dir = "/home/dchenbs/workspace/datasets/sa1b"
    pseudo_dir = "/home/dchenbs/workspace/datasets/sa1b_directsam/res_1800px_thr.03_output" 
    output_dir = "/home/dchenbs/workspace/datasets/sa1b_directsam/res_1800px_thr.03_output_merged" 

    thickness = 3
    resolution = 1800

    os.makedirs(output_dir, exist_ok=True)

    # list all json files in output_dir including subdirectories
    image_paths = []
    json_files = []
    for root, dirs, files in os.walk(pseudo_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                json_files.append(json_file)
                image_paths.append(json_file.replace(pseudo_dir, sa1b_dir).replace(".json", ".jpg"))

    print(f"Found {len(json_files)} pseudo labels")

    tasks = [(json_files[i], image_paths[i], image_paths[i].replace(".jpg", ".json"), 
              resolution, thickness, pseudo_dir, output_dir) for i in range(len(json_files))]

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            pass

if __name__ == "__main__":
    main()