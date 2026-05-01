import numpy as np
import lap

def intersection_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    return w * h

def iou_batch(bboxes1, bboxes2):
    bboxes1 = np.ascontiguousarray(bboxes1)
    bboxes2 = np.ascontiguousarray(bboxes2)
    if bboxes1.ndim == 1: bboxes1 = bboxes1[None, :]
    if bboxes2.ndim == 1: bboxes2 = bboxes2[None, :]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    inter = intersection_batch(bboxes1, bboxes2)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union

def fish_iou_batch(bboxes1, bboxes2):
    """ Specialized IoU for fish tracking considering scale-awareness (Shape Consistency) """
    iou = iou_batch(bboxes1, bboxes2)
    # Scale-aware refinement (Fish-specific morphology)
    w1, h1 = bboxes1[:, 2] - bboxes1[:, 0], bboxes1[:, 3] - bboxes1[:, 1]
    w2, h2 = bboxes2[:, 2] - bboxes2[:, 0], bboxes2[:, 3] - bboxes2[:, 1]
    
    # Aspect ratio comparison
    r1 = w1 / (h1 + 1e-6)
    r2 = w2 / (h2 + 1e-6)
    ratio_diff = np.abs(r1[:, None] - r2[None, :])
    
    # Apply penalty for shape changes (prevents ID switches when boxes deform)
    return iou * (1 - 0.1 * ratio_diff)

def linear_assignment(cost_matrix):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])
