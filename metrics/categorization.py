def accuracy(pred_ans: str, gt_ans: str) -> float:
    return float(pred_ans==gt_ans)