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

    def __call__(self, image):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.model.device).to(self.model.dtype)
        logits = self.model(pixel_values=pixel_values).logits.float().cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(self.resolution, self.resolution),
            mode="bicubic",
        )
        probabilities = torch.sigmoid(upsampled_logits)[0,0]
        boundary = (probabilities > self.threshold).detach().numpy()
        boundary, num_tokens = self.boundary_post_processing(boundary)

        return boundary, num_tokens

    def boundary_post_processing(self, boundary):

        boundary = skeletonize(boundary)
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

