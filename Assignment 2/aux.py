import torch
from torch import nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from torchvision.transforms import ToTensor, Lambda
from torchvision.io import decode_image
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead

from torchtune.datasets import ConcatDataset

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from tqdm import tqdm

import os
import glob


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def plot_images(image, boxes, title = None):
    _, height, width = img.shape

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img.permute(1, 2, 0).cpu())
    ax.set_title(title)
    ax.axis("off")

    for i, box in enumerate(boxes):
        cx, cy, w, h = box[2:6]
        
        class_name = class_names[class_index]
        
        x1, y1, x2, y2 = box[:4]
        rect = patches.Rectangle(
            (cx - w/2, cy - h/2), 
            cx + w/2, cy + h/2, 
            linewidth = 2, edgecolor = 'r', facecolor = 'none'
        )
        ax.add_patch(rect)
        
        # Add class and confidence
        class_index = box[0]
        class_name = classes[class_index]
        conf = box[1]
        ax.text(
            cx, cy - 5, 
            f"{class_name}: {conf:.2f}", 
            color = "white", fontsize = 10, backgroundcolor = "red"
        )
    plt.show()
