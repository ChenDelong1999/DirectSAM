import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize



# def recall_with_tolerance(target, prediction, tolerance):

#     if np.sum(target) == 0:
#         return 1.0

#     # Boundary zero padding (BZP)
#     target[:tolerance, :] = target[-tolerance:, :] = target[:, :tolerance] = target[:, -tolerance:] = 0
#     prediction[:tolerance, :] = prediction[-tolerance:, :] = prediction[:, :tolerance] = prediction[:, -tolerance:] = 0

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2, tolerance*2))
#     prediction_blured = cv2.filter2D(prediction.astype(np.float32), -1, kernel) > 0
#     # prediction_blured = cv2.GaussianBlur(prediction.astype(np.float32), (tolerance, tolerance), 0) > 0

#     true_positive = np.sum(target * prediction_blured)
#     recall = true_positive / np.sum(target)
#     return recall if not np.isnan(recall) else 0.0


def effective_contour_recall(gt_contour, pred_contour, tolerance=10, do_visualization=False):

    gt_contour[:tolerance, :] = gt_contour[-tolerance:, :] = gt_contour[:, :tolerance] = gt_contour[:, -tolerance:] = 0
    pred_contour[:tolerance, :] = pred_contour[-tolerance:, :] = pred_contour[:, :tolerance] = pred_contour[:, -tolerance:] = 1

    if np.sum(gt_contour) == 0:
        return -1, -1

    pred_contour_skeletonized = skeletonize(pred_contour)
    num_tokens, pred_label_map = cv2.connectedComponents(1-pred_contour_skeletonized.astype(np.uint8), connectivity=4)
    effective_contour = np.zeros_like(gt_contour).astype(np.uint8)

    kernel_contour = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for label_id in range(1, num_tokens):
        mask = pred_label_map == label_id
        
        dialated_mask = cv2.dilate(mask.astype(np.uint8), kernel_contour, iterations=1)
        erosioned_mask = cv2.erode(dialated_mask, kernel_contour, iterations=1)

        contour = dialated_mask - erosioned_mask
        effective_contour[contour > 0] = 1

    kernel_tolerance = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance, tolerance))
    effective_contour_dilated = cv2.dilate(effective_contour, kernel_tolerance, iterations=1)

    recall = np.sum(np.logical_and(gt_contour, effective_contour_dilated)) / np.sum(gt_contour)

    if do_visualization:

        plt.subplot(1, 5, 2)
        plt.title('Predicted Contour')
        plt.imshow(pred_contour, cmap='binary')
        plt.axis('off')

        plt.subplot(1, 5, 3)
        plt.title(f'Effective Contour ({num_tokens-1} Tokens)')
        plt.imshow(effective_contour_dilated, cmap='binary')
        plt.axis('off')

        overlapping = np.logical_and(gt_contour, effective_contour_dilated)
        missing = np.logical_and(gt_contour, np.logical_not(effective_contour_dilated))

        overlapping = cv2.dilate(overlapping.astype(np.uint8), kernel_contour, iterations=1) > 0
        missing = cv2.dilate(missing.astype(np.uint8), kernel_contour, iterations=1) > 0

        canvas = np.ones((gt_contour.shape[0], gt_contour.shape[1], 3), dtype=np.uint8) * 255
        canvas[overlapping] = [0, 0, 0]
        canvas[missing] = [255, 0, 0]

        plt.subplot(1, 5, 4)
        plt.title(f'Overlapping & Missing Contours (Recall: {recall:.2f})')
        plt.imshow(canvas)
        plt.axis('off')


        plt.subplot(1, 5, 5)
        plt.title('Ground Truth Contour')
        plt.imshow(gt_contour, cmap='binary')
        plt.axis('off')

    return num_tokens-1, recall


