import torch
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
import torch.nn as nn


class DirectSAM():

    def __init__(self, model_name, resolution, device):
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device).half().eval()
        self.processor = AutoImageProcessor.from_pretrained('chendelong/DirectSAM-1800px-0424')

        self.processor.size['height'] = resolution
        self.processor.size['width'] = resolution
        self.resolution = resolution

    def __call__(self, image):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.model.device).to(self.model.dtype)
        logits = self.model(pixel_values=pixel_values).logits.float().cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(self.resolution, self.resolution),
            mode="bicubic",
        )
        probabilities = torch.sigmoid(upsampled_logits).detach().numpy()[0,0]
        return probabilities
