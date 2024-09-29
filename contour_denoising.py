import os
import json
import numpy as np
from pycocotools.mask import encode, decode
from tqdm import tqdm
import cv2
from model.contour_denoiser import contour_denoising

def process_file(args):
    input_json_file, output_json_file, resolution = args
    try:
        if os.path.exists(output_json_file):
            return

        # Load the RLE data from the JSON file
        with open(input_json_file, "r") as f:
            data = json.load(f)

        # Decode the RLE to get the mask
        mask = decode(data).squeeze().astype(np.uint8)
        # Resize the mask to the desired resolution
        mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)

        # Apply contour denoising
        denoised_mask = contour_denoising(mask, skip_merging=4, radius=2, area_ratio=1/5000)
        denoised_mask = denoised_mask.astype(np.uint8)

        # Encode the denoised mask back to RLE
        rle = encode(np.asfortranarray(denoised_mask))
        rle['counts'] = rle['counts'].decode('utf-8')

        # Save the new RLE to the output JSON file
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
        with open(output_json_file, 'w') as f:
            json.dump(rle, f, indent=4)

    except Exception as e:
        print(f"Error processing {input_json_file}: {e}")

def main():
    input_dir = "/home/dchenbs/workspace/datasets/sa1b_directsam/res_1800px_thr.03_output_merged"
    output_dir = "/home/dchenbs/workspace/datasets/sa1b_directsam/res_1800px_thr.03_output_merged_denoised"

    resolution = 768

    os.makedirs(output_dir, exist_ok=True)

    # Gather all JSON files from input_dir
    input_json_files = []
    output_json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                input_json_file = os.path.join(root, file)
                output_json_file = input_json_file.replace(input_dir, output_dir)
                input_json_files.append(input_json_file)
                output_json_files.append(output_json_file)

    part_idx = 7
    per_part = 200000

    # print(f"Total {len(input_json_files)} files")
    # print(f"Processing {part_idx}th part ({part_idx * per_part}-{len(input_json_files)}")
    
    for i in tqdm(range(len(input_json_files))):
    # for i in tqdm(range(part_idx * per_part, len(input_json_files))):
        process_file((input_json_files[i], output_json_files[i], resolution))


if __name__ == "__main__":
    main()