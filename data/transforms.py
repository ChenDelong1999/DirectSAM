import cv2
import json
import numpy as np
from PIL import Image as PILImage
from pycocotools.mask import decode, frPyObjects
from skimage.morphology import skeletonize


def boundary_thinning(boundary, thickness=2):
    if thickness > 1:
        boundary = skeletonize(boundary).astype(np.uint8)
        boundary = cv2.dilate(boundary, np.ones((thickness, thickness), np.uint8))
    return boundary


def resize_image(image, resolution):
    if isinstance(image, PILImage.Image):
        return image.resize((resolution, resolution))
    elif type(image) == np.ndarray:
        return cv2.resize(image, (resolution, resolution))
    else:
        raise ValueError(f"Unknown input type: {type(image)}")

import cv2
import numpy as np

import cv2
import numpy as np

# Initialize the round kernel once
def create_circular_kernel(size):
    kernel = np.zeros((size, size), np.uint8)
    center = size // 2
    cv2.circle(kernel, (center, center), center, 1, -1)
    return kernel

# Global circular kernel
CIRCULAR_KERNEL = create_circular_kernel(3)


def preprocess_label_map(label_map, resolution, label_map_mode=''):

    label_map = np.array(label_map)
    
    if label_map_mode=='single_channel':
        pass

    elif label_map_mode=='force_binary':
        threshold = label_map.max() / 2
        label_map = (label_map > threshold).astype(np.uint8)

    elif label_map_mode=='rgb':
        label_map = cv2.cvtColor(label_map, cv2.COLOR_RGB2GRAY)

    else:
        raise ValueError(f"Unknown label_map_mode: {label_map_mode}")


    if len(label_map.shape) == 3:
        label_map = label_map[:, :, 0]
        
    label_map = cv2.resize(
        label_map, (resolution, resolution), 
        interpolation=cv2.INTER_NEAREST
        )

    return label_map


def label_map_to_boundary(label_map, thickness=3):

    global CIRCULAR_KERNEL
    
    if thickness != CIRCULAR_KERNEL.shape[0]:
        CIRCULAR_KERNEL = create_circular_kernel(thickness)

    unique_labels = np.unique(label_map)
    masks = (label_map[..., np.newaxis] == unique_labels).astype(np.uint8)

    dilated = cv2.dilate(masks, CIRCULAR_KERNEL)
    eroded = cv2.erode(masks, CIRCULAR_KERNEL)
    boundaries = dilated - eroded

    if len(boundaries.shape) == 2:
        boundaries = boundaries[..., np.newaxis]
    boundaries = np.any(boundaries, axis=-1).astype(bool)

    boundaries[:thickness, :] = boundaries[-thickness:, :] = 1
    boundaries[:, :thickness] = boundaries[:, -thickness:] = 1

    return boundaries



def transforms_for_labelmap_dataset(batch, resolution, thickness, label_map_mode='single_channel'):
    image = [resize_image(x.convert("RGB"), resolution) for x in batch["image"]]
    label_map = [preprocess_label_map(x, resolution, label_map_mode) for x in batch["annotation"]]
    labels = [label_map_to_boundary(x, thickness) for x in label_map]
    return {'image': image, 'label': labels, 'label_map': label_map}

