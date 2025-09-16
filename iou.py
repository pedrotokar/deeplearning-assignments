# [label, cx, cy, w, h]

import numpy as np

def iou(box1, box2):
    b1_x1 = box1[1] - box1[3]/2
    b1_x2 = box1[1] + box1[3]/2
    b1_y1 = box1[2] - box1[4]/2
    b1_y2 = box1[2] + box1[4]/2

    b2_x1 = box2[1] - box2[3]/2
    b2_x2 = box2[1] + box2[3]/2
    b2_y1 = box2[2] - box2[4]/2
    b2_y2 = box2[2] + box2[4]/2

    x_overlap = max(0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1))
    y_overlap = max(0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    
    intersection = x_overlap * y_overlap
    
    union = box1[3] * box1[4] + box2[3] * box2[4] - intersection
    
    return intersection / union