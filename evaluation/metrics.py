import numpy as np
import cv2
import torch
from torch.nn import functional as F


def edge_zero_padding(boundary, offset):
    boundary[:offset, :] = 0
    boundary[-offset:, :] = 0
    boundary[:, :offset] = 0
    boundary[:, -offset:] = 0
    return boundary


def calculate_recall_torch(target, predictions, r, device):

    r = int(r)
    assert r % 2 == 1 and r > 0
    target = torch.tensor(target).to(device).float()
    predictions = torch.tensor(predictions).to(device).float()

    C, H, W = predictions.shape
    kernel = torch.ones((C, 1, r, r)).to(device)

    predictions_blur = F.conv2d(predictions, kernel, groups=C, padding=r//2)
    overlap = target * (predictions_blur > 0)
    recall = overlap.sum(dim=(1, 2)) / target.sum()

    return recall.cpu().numpy()


def get_num_tokens(boundary):
    num_objects, labels = cv2.connectedComponents((1-boundary).astype(np.uint8))
    return num_objects


def get_metrics(target, prob, thresholds, bzp_offset, tolerance, device):
    target = edge_zero_padding(target, bzp_offset)

    predictions = np.array([prob > threshold for threshold in thresholds])

    all_num_tokens = []
    for prediction in predictions:
        num_tokens = get_num_tokens(prediction)
        all_num_tokens.append(num_tokens)

    all_recall = calculate_recall_torch(target, predictions, tolerance, device)

    return all_num_tokens, all_recall