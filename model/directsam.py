import torch
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
import torch.nn as nn
import numpy as np
import cv2
from skimage.morphology import skeletonize

class DirectSAM():

    def __init__(self, model_name, resolution, threshold=0.5, device='cuda'):
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device).half().eval()
        self.processor = AutoImageProcessor.from_pretrained('chendelong/DirectSAM-1800px-0424')

        self.processor.size['height'] = resolution
        self.processor.size['width'] = resolution
        self.resolution = resolution
        self.threshold = threshold

    def __call__(self, image, post_processing=True):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.model.device).to(self.model.dtype)
        logits = self.model(pixel_values=pixel_values).logits.float().cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(self.resolution, self.resolution),
            mode="bicubic",
        )
        probabilities = torch.sigmoid(upsampled_logits)[0,0].detach().numpy()
        
        
        if post_processing:
            boundary, num_tokens = self.boundary_post_processing(probabilities)
        else:
            boundary = probabilities > self.threshold
            num_tokens = -1

        return boundary, num_tokens

    def boundary_post_processing(self, probabilities):

        kernel_size = self.resolution // 200
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.float32)
        kernel /= kernel.sum()
        
        probabilities = cv2.filter2D(probabilities, -1, kernel) 

        boundary = probabilities > self.threshold
        boundary = skeletonize(boundary)

        boundary[:kernel_size, :] = boundary[-kernel_size:, :] = boundary[:, :kernel_size] = boundary[:, -kernel_size:] = 1

        num_objects, labels = cv2.connectedComponents(
            (1-boundary).astype(np.uint8), 
            connectivity=4, 
            )
        
        valid_boundary = np.zeros_like(boundary).astype(np.uint8)
        for i in range(1, num_objects):
            mask = labels == i
            
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

            dialated = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8))
            eroded = cv2.erode(mask, np.ones((2, 2), np.uint8))

            valid_boundary += (dialated - eroded)
        return valid_boundary > 0, num_objects - 1

