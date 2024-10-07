import cv2
import numpy as np
from PIL import Image as PILImage
from skimage.morphology import skeletonize
from torchvision import transforms
import torch


def boundary_thinning(boundary, thickness=2):
    if thickness > 1:
        boundary = skeletonize(boundary).astype(np.uint8)
        boundary = cv2.dilate(boundary, np.ones((thickness, thickness), np.uint8))
    return boundary


augmentation = transforms.RandomApply(torch.nn.ModuleList([
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomGrayscale(p=0.2),
    ]), p=0.25)


def preprocess_image(image, resolution, do_augmentation):
    if type(image) == np.ndarray:
        image = PILImage.fromarray(image)
    
    image = image.resize((resolution, resolution))

    if do_augmentation:
        return augmentation(image)
    else:
        return image


# Initialize the round kernel once
def create_circular_kernel(size):
    if size > 3:
        kernel = np.zeros((size, size), np.uint8)
        center = size // 2
        cv2.circle(kernel, (center, center), center, 1, -1)
    else:
        kernel = np.ones((size, size), np.uint8)
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



def transforms_for_labelmap_dataset(batch, resolution, thickness, image_key="image", annotation_key="annotation", label_map_mode='single_channel', do_augmentation=False, **kwargs):
    image = [preprocess_image(x.convert("RGB"), resolution, do_augmentation) for x in batch[image_key]]
    label_map = [preprocess_label_map(x, resolution, label_map_mode) for x in batch[annotation_key]]
    labels = [label_map_to_boundary(x, thickness) for x in label_map]
    return {'image': image, 'label': labels, 'label_map': label_map}



def transforms_for_contour_dataset(batch, resolution, image_key="image", annotation_key="annotation", do_augmentation=False, **kwargs):
    image = [preprocess_image(x.convert("RGB"), resolution, do_augmentation) for x in batch[image_key]]
    labels = [np.array(x.resize((resolution, resolution))).astype(bool) for x in batch[annotation_key]]

    return {'image': image, 'label': labels, 'label_map': labels}

