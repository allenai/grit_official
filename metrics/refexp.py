import numpy as np
from metrics.localization import compute_iou


def refexp_metric(pred_boxes: list[list], gt_boxes: list[list]) -> float:
    if len(pred_boxes) == 0:
        return 0.0
    
    # select the first box
    pred_box = pred_boxes[0]
    gt_box = gt_boxes[0]
    
    iou = compute_iou(pred_box,gt_box)

    return float(iou > 0.5)