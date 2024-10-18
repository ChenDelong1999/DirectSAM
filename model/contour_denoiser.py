import torch
from skimage.morphology import skeletonize
import numpy as np
import cv2

class ContourDenoiser:
    def __init__(self, max_tokens=128, area_threshold=0, device='cuda', radius=5):
        """
        Initializes the ContourDenoiser with the given parameters.

        Args:
            max_tokens (int): Maximum number of masks allowed.
            area_threshold (float): Area threshold below which masks are considered small.
            device (str): Device to perform computations.
            radius (int): Radius for morphological operations.
        """
        self.max_tokens = max_tokens
        self.area_threshold = area_threshold
        self.device = device
        self.radius = radius

        # Initialize the circular kernel once
        self.kernel = self.create_circular_kernel(radius=self.radius, device=self.device)

    @staticmethod
    @torch.no_grad()
    def create_circular_kernel(radius=5, device='cuda'):
        """Creates a circular kernel for morphological operations."""
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel_array = (x**2 + y**2 <= radius**2).astype(np.float32)
        kernel = torch.tensor(kernel_array, device=device).unsqueeze(0).unsqueeze(0)
        return kernel

    @torch.no_grad()
    def dilate_masks(self, masks):
        """Dilates binary masks using the pre-initialized circular kernel."""
        masks = masks.unsqueeze(1).float()  # Add channel dimension
        dilated = torch.nn.functional.conv2d(masks, self.kernel, padding=self.radius)
        dilated = (dilated > 0.5).float()
        return dilated.squeeze(1)  # Remove channel dimension

    @torch.no_grad()
    def erode_masks(self, masks):
        """Erodes binary masks using the pre-initialized circular kernel."""
        masks = masks.unsqueeze(1).float()
        inverted_masks = 1 - masks
        eroded_inverted = torch.nn.functional.conv2d(inverted_masks, self.kernel, padding=self.radius)
        eroded = 1 - eroded_inverted
        eroded = (eroded > 0.5).float()
        return eroded.squeeze(1)

    @torch.no_grad()
    def merge_masks(self, labels):
        """
        Merges small masks into larger neighboring masks based on area threshold and limits the total number of masks.

        Args:
            labels (torch.Tensor): Tensor of labels with shape (H, W).

        Returns:
            torch.Tensor: Tensor of merged masks.
        """
        device = self.device
        area_threshold = self.area_threshold
        max_tokens = self.max_tokens
        radius = self.radius

        max_label = labels.max().item()
        label_indices = torch.arange(1, max_label + 1, device=device)
        masks = (labels.unsqueeze(0) == label_indices.view(-1, 1, 1)).float()
        areas = masks.sum(dim=(1, 2))

        # Identify small and large masks
        if area_threshold > 0:
            small_mask_indices = (areas < area_threshold).nonzero(as_tuple=True)[0]
            large_mask_indices = (areas >= area_threshold).nonzero(as_tuple=True)[0]
        else:
            small_mask_indices = torch.tensor([], dtype=torch.long, device=device)
            large_mask_indices = torch.arange(len(areas), device=device)

        # Initialize label mapping
        label_mapping = torch.zeros(max_label + 1, dtype=torch.long, device=device)
        label_mapping[0] = 0  # Background label

        # Assign labels to large masks
        if large_mask_indices.numel() > 0:
            label_mapping[large_mask_indices + 1] = torch.arange(1, large_mask_indices.numel() + 1, device=device)

        # Merge small masks into neighboring large masks
        if small_mask_indices.numel() > 0 and large_mask_indices.numel() > 0:
            small_masks = masks[small_mask_indices]
            large_masks = masks[large_mask_indices]

            # Dilate all small masks together
            dilated_small_masks = self.dilate_masks(small_masks)

            # Compute overlaps
            overlaps = torch.einsum('shw,lhw->sl', dilated_small_masks, large_masks)
            overlap_max_values, best_matches = overlaps.max(dim=1)

            # Update label mapping
            matched_indices = overlap_max_values > 0
            matched_small_indices = small_mask_indices[matched_indices]
            matched_large_indices = large_mask_indices[best_matches[matched_indices]]
            label_mapping[matched_small_indices + 1] = label_mapping[matched_large_indices + 1]

            # Discard unmatched small masks
            unmatched_small_indices = small_mask_indices[~matched_indices]
            label_mapping[unmatched_small_indices + 1] = 0
        else:
            # Discard small masks if no large masks are present
            label_mapping[small_mask_indices + 1] = 0

        # Apply new labels
        new_labels = label_mapping[labels]

        # Limit the number of masks to max_tokens
        unique_labels, counts = torch.unique(new_labels, return_counts=True)
        valid_indices = unique_labels != 0
        unique_labels = unique_labels[valid_indices]
        counts = counts[valid_indices]

        if len(unique_labels) > max_tokens:
            # Merge smaller masks to limit the number of tokens
            sorted_indices = torch.argsort(counts)
            labels_to_keep = unique_labels[sorted_indices][-max_tokens:]
            labels_to_merge = unique_labels[sorted_indices][:-max_tokens]

            # Initialize final label mapping
            final_label_mapping = torch.zeros_like(label_mapping)
            final_label_mapping[labels_to_keep] = torch.arange(1, len(labels_to_keep) + 1, device=device)

            # Prepare masks for merging
            masks_to_merge = (new_labels.unsqueeze(0) == labels_to_merge.view(-1, 1, 1)).float()
            masks_to_keep = (new_labels.unsqueeze(0) == labels_to_keep.view(-1, 1, 1)).float()

            # Dilate masks to merge
            dilated_masks_to_merge = self.dilate_masks(masks_to_merge)
            overlaps = torch.einsum('mhw,khw->mk', dilated_masks_to_merge, masks_to_keep)
            overlap_max_values, best_matches = overlaps.max(dim=1)

            # Update final label mapping
            matched_indices = overlap_max_values > 0
            matched_merge_labels = labels_to_merge[matched_indices]
            matched_keep_labels = labels_to_keep[best_matches[matched_indices]]
            final_label_mapping[matched_merge_labels] = final_label_mapping[matched_keep_labels]
            unmatched_merge_labels = labels_to_merge[~matched_indices]
            final_label_mapping[unmatched_merge_labels] = 0

            new_labels = final_label_mapping[new_labels]
        else:
            # Adjust labels to be sequential
            final_label_mapping = torch.zeros_like(label_mapping)
            final_label_mapping[unique_labels] = torch.arange(1, len(unique_labels) + 1, device=device)
            new_labels = final_label_mapping[new_labels]

        # Generate new masks
        unique_labels = torch.unique(new_labels)
        unique_labels = unique_labels[unique_labels != 0]
        new_masks = (new_labels.unsqueeze(0) == unique_labels.view(-1, 1, 1)).float()
        return new_masks

    @torch.no_grad()
    def contour_denoising(self, contour):
        """
        Denoises the contour and segments it into meaningful regions.

        Args:
            contour (numpy.ndarray): Input contour.

        Returns:
            numpy.ndarray: Denoised contour.
        """
        device = self.device
        radius = self.radius

        # Convert contour to tensor and dilate to fill gaps
        contour_tensor = torch.tensor(contour, device=device, dtype=torch.float32)
        dilated_contour = self.dilate_masks(contour_tensor.unsqueeze(0))[0].cpu().numpy()

        # Skeletonize the dilated contour
        skeletonized_contour = skeletonize(dilated_contour > 0)

        # Find connected components
        _, labels = cv2.connectedComponents(1 - skeletonized_contour.astype(np.uint8), connectivity=4)
        labels = torch.tensor(labels, device=device, dtype=torch.long)

        # Merge masks
        masks = self.merge_masks(labels)

        # Refine masks
        masks = self.dilate_masks(masks)
        eroded_masks = self.erode_masks(masks)
        refined_contour = masks - eroded_masks
        refined_contour = (refined_contour.sum(dim=0) > 0.5).float()

        # Include borders in the contour
        refined_contour[:radius*2, :] = 1
        refined_contour[-radius*2:, :] = 1
        refined_contour[:, :radius*2] = 1
        refined_contour[:, -radius*2:] = 1

        return refined_contour.cpu().numpy()

