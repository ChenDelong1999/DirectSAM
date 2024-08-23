import cv2
import json
import numpy as np
from PIL import Image as PILImage
from pycocotools.mask import decode, frPyObjects
from skimage.morphology import skeletonize


def boundary_thinning(boundary, thickness=2):
    boundary = skeletonize(boundary).astype(np.uint8)
    if thickness > 1:
        boundary = cv2.dilate(boundary, np.ones((thickness, thickness), np.uint8))
    return boundary


def resize_image(image, resolution):
    if isinstance(image, PILImage.Image):
        return image.resize((resolution, resolution))
    elif type(image) == np.ndarray:
        return cv2.resize(image, (resolution, resolution))
    else:
        raise ValueError(f"Unknown input type: {type(image)}")


def masks_to_boundary(masks, thickness=3):
    """
    Parameters:
    masks (numpy array): The input masks. Shape: (num_masks, height, width).
    thickness (int): The size of the kernel used for dilation and erosion.

    Returns:
    numpy array: The output binary boundary label image. Shape: (height, width).
    """
    kernel = np.ones((thickness, thickness), np.uint8)

    masks = masks.astype(np.uint8)

    dilated = np.array([cv2.dilate(mask, kernel) for mask in masks])
    eroded = np.array([cv2.erode(mask, kernel) for mask in masks])

    boundaries = dilated - eroded
    boundaries = np.sum(boundaries, axis=0)

    return boundaries > 0


def annotation_to_label(label_map, resolution, thickness=3):
    """
    Parameters:
    label_map (PIL.Image or numpy array): The input label map (single channel).
    resolution (int): The resolution of the output label map.
    thickness (int): The size of the kernel used for dilation and erosion.

    Returns:
    PIL.Image: The output binary boundary label image.
    """

    label_map = np.array(label_map)

    # avoid using PIL.Image.convert("L") 
    if len(label_map.shape) == 3:
        label_map = label_map[:, :, 0]

    masks = []
    for label_idx in np.unique(label_map):
        mask = (label_map == label_idx).astype(np.uint8)
        mask = cv2.resize(mask, (resolution, resolution))

        masks.append(mask)

    masks = np.array(masks)
    boundary = masks_to_boundary(masks, thickness)

    return boundary


def transforms_huggingface_dataset(example_batch, resolution, thickness):
    images = [resize_image(x.convert("RGB"), resolution) for x in example_batch["image"]]
    labels = [annotation_to_label(x, resolution, thickness) for x in example_batch["annotation"]]
    return {'image': images, 'label': labels}


def transforms_image_folders(example_batch, resolution, thickness, image_suffix='.jpg', label_suffix='.png'):
    images = [resize_image(PILImage.open(x+image_suffix).convert("RGB"), resolution) for x in example_batch["image"]]
    labels = [annotation_to_label(PILImage.open(x+label_suffix), resolution, thickness) for x in example_batch["label"]]
    return {'image': images, 'label': labels}


def transforms_coco_single_sample(image, annotations, resolution, thickness):
    masks = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    for i, annotation in enumerate(annotations):
        rles = frPyObjects(annotation['segmentation'], image.size[1], image.size[0])
        mask = decode(rles)
        if len(mask.shape) == 3:
            mask = mask.sum(axis=2)
        masks += mask * (i + 1)
    label = annotation_to_label(masks, resolution, thickness)
    image = resize_image(image, resolution)
    return image, label


def transforms_entity_seg(example_batch, resolution, thickness):

    def decode_and_merge_rle_annotations(annotations):
        masks = decode(annotations[0])
        for i in range(1, len(annotations)):
            masks += decode(annotations[i])*(i+1)
        return masks

    images = [resize_image(PILImage.open(x), resolution) for x in example_batch["image"]]
    labels = [annotation_to_label(decode_and_merge_rle_annotations(x), resolution, thickness) for x in example_batch["label"]]

    return {'image': images, 'label': labels}


def transforms_directsam_pseudo_label(example_batch, resolution, thickness):
    images = [resize_image(PILImage.open(x), resolution) for x in example_batch["image"]]
    # labels = [resize_image(decode(json.load(open(x))), resolution) for x in example_batch["label"]]
    labels = [boundary_thinning(resize_image(decode(json.load(open(x))), resolution), thickness) for x in example_batch["label"]]

    return {'image': images, 'label': labels}

