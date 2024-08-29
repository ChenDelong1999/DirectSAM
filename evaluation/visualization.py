import cv2
import numpy as np


def compare_boundaries(target, prediction, tolerance, linewidth, brightness=150):

    target_blured = cv2.GaussianBlur(target.astype(np.float32), (tolerance, tolerance), 0) > 0
    prediction_blured = cv2.GaussianBlur(prediction.astype(np.float32), (tolerance, tolerance), 0) > 0

    gray = target * prediction_blured
    red = target * (prediction_blured == 0)
    blue = prediction * (target_blured == 0)

    gray = cv2.GaussianBlur(gray.astype(np.float32), (linewidth, linewidth), 0) > 0
    red = cv2.GaussianBlur(red.astype(np.float32), (linewidth, linewidth), 0) > 0
    blue = cv2.GaussianBlur(blue.astype(np.float32), (linewidth, linewidth), 0) > 0

    image = np.ones((target.shape[0], target.shape[1], 3)) * 255
    image[gray] = [brightness, brightness, brightness]
    image[red] = [brightness, 0, 0]
    image[blue] = [0, 0, brightness]

    image[:tolerance, :, :] = image[-tolerance:, :, :] = image[:, :tolerance, :] = image[:, -tolerance:, :] = 255

    return image.astype(np.uint8)

