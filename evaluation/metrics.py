import numpy as np
import cv2


def recall_with_tolerance(target, prediction, tolerance):

    if np.sum(target) == 0:
        return 1.0

    # Boundary zero padding (BZP)
    target[:tolerance, :] = target[-tolerance:, :] = target[:, :tolerance] = target[:, -tolerance:] = 0
    prediction[:tolerance, :] = prediction[-tolerance:, :] = prediction[:, :tolerance] = prediction[:, -tolerance:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2, tolerance*2))
    prediction_blured = cv2.filter2D(prediction.astype(np.float32), -1, kernel) > 0
    # prediction_blured = cv2.GaussianBlur(prediction.astype(np.float32), (tolerance, tolerance), 0) > 0

    true_positive = np.sum(target * prediction_blured)

    recall = true_positive / np.sum(target)

    return recall if not np.isnan(recall) else 0.0