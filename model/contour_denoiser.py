




import torch
from skimage.morphology import skeletonize
import numpy as np
import cv2
import matplotlib.pyplot as plt

@torch.no_grad()
def get_kernel(radius=5, device='cuda'):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device).float()
    return kernel

@torch.no_grad()
def dilate(masks, radius=5):
    dtype = masks.dtype
    kernel = get_kernel(radius)
    masks = masks.unsqueeze(1)
    dilated_masks = torch.nn.functional.conv2d(masks.float(), kernel, padding=radius)
    binary_masks = (dilated_masks > 0.5).float()
    return binary_masks.squeeze(1).to(dtype)

@torch.no_grad()
def erode(masks, radius=5):
    kernel = get_kernel(radius)
    masks = masks.unsqueeze(1)

    inverted_masks = 1 - masks.float()
    dilated_inverted = torch.nn.functional.conv2d(inverted_masks, kernel, padding=radius)

    eroded_masks = 1 - dilated_inverted
    binary_masks = (eroded_masks > 0.5).float()
    
    return binary_masks.squeeze(1)

def contour_to_mask(boundary):
    """
    Converts a boundary image to a binary mask.
    Input:      A numpy array (H, W) representing the boundary image, True for boundary pixels and False for non-boundary pixels.
    Returns:    A numpy array of binary masks (n_masks, H, W), where each mask corresponds to a connected component in the boundary image.
    """
    num_objects, labels = cv2.connectedComponents(
        1-boundary.astype(np.uint8), 
        connectivity=4, 
        )

    masks = np.zeros((num_objects-1, *boundary.shape), dtype=bool)
    for i in range(1, num_objects):
        masks[i-1] = labels == i

    # sort by area
    areas = np.sum(masks, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::-1]
    masks = masks[sorted_indices]

    return masks

def merge_masks(labels, threshold, device='cuda'):
    """
    Merges small masks into neighboring larger masks based on the strongest overlap.

    Small masks are assigned to the large mask with which they have the largest overlap after dilation.
    Small masks that have no neighboring large masks are discarded.

    Args:
        labels (torch.Tensor): A tensor of label indices where each connected component is assigned a unique integer label.
                               The background is represented by 0.
        threshold (float): The area threshold below which masks are considered small and need to be merged.

    Returns:
        torch.Tensor: A tensor of merged masks where small masks have been assigned to large masks based on strongest overlap.
                      Masks that are smaller than the threshold and have no neighboring large masks are discarded.
    """
    device = labels.device
    max_label = labels.max().item()
    
    # Create binary masks for each label
    masks = (labels.unsqueeze(0) == torch.arange(1, max_label + 1, device=device).unsqueeze(1).unsqueeze(2))
    # Shape: (num_masks, H, W)
    areas = masks.float().sum(dim=(1, 2))
    mask_indices = torch.arange(len(masks), device=device)
    
    # Identify small and large masks
    small_mask_indices = mask_indices[areas < threshold]
    large_mask_indices = mask_indices[areas >= threshold]
    
    small_masks = masks[small_mask_indices]
    large_masks = masks[large_mask_indices]
    large_labels = large_mask_indices + 1  # Adjust for labels starting from 1
    
    # Prepare label mapping
    label_mapping = torch.arange(0, max_label + 1, device=device)
    
    # For each small mask, compute overlap with large masks
    for i, small_mask in enumerate(small_masks):
        small_label = small_mask_indices[i] + 1  # Labels start from 1
        small_mask_dilated = dilate(small_mask.unsqueeze(0).float(), radius=5)[0]
        # Compute overlaps with large masks
        overlaps = (small_mask_dilated.unsqueeze(0) * large_masks.float()).sum(dim=(1, 2))  # Shape: (num_large_masks,)
        if overlaps.max() > 0:
            max_idx = overlaps.argmax()
            target_label = large_labels[max_idx]
            label_mapping[small_label] = target_label
        else:
            # Discard small mask
            label_mapping[small_label] = 0  # Map to background

    # Apply label mapping
    new_labels = label_mapping[labels]
    
    # Reconstruct masks from new labels
    unique_labels = torch.unique(new_labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background label 0
    
    new_masks = (new_labels.unsqueeze(0) == unique_labels.unsqueeze(1).unsqueeze(2)).float()
    
    return new_masks



def contour_denoising(contour, device='cuda', skip_merging=4, radius=3, area_ratio=1/1000):

    contour = skeletonize(contour>0)
    # contour[:1, :] = contour[-1:, :] = 1
    # contour[:, :1] = contour[:, -1:] = 1
    

    num_objects, labels = cv2.connectedComponents(
        1-contour.astype(np.uint8), 
        connectivity=4, 
    )
    labels = torch.tensor(labels).to(device)

    masks = torch.stack([labels == i for i in range(1, num_objects)], dim=0).float()

    small_segment_threshold = area_ratio * contour.shape[0] * contour.shape[1]
    if num_objects > skip_merging:
        masks = merge_masks(labels, threshold=small_segment_threshold)

    masks = dilate(masks, radius=radius)

    eroded_masks = erode(masks, radius=radius)
    contour = masks - eroded_masks
    contour = torch.sum(contour, dim=0, keepdim=True)[0] > 0.5

    contour[:radius*2, :] = contour[-radius*2:, :] = 1
    contour[:, :radius*2] = contour[:, -radius*2:] = 1
    
    return contour.cpu().numpy()
 