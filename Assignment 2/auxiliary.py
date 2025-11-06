#%%
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

#from torchtune.datasets import ConcatDataset

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

def plot_images(dataset, index, title = None):
    #TODO: ADICIONAR PREVISÃO DO MODELO NO PLOT
    image = dataset.get_plot_image(index)
    _, height, width = image.shape

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image.permute(1, 2, 0).cpu())
    ax.set_title(title)
    #ax.axis("off")

    boxes = dataset.get_plot_box(index)
    for i, box in enumerate(boxes):
        class_index = int(box[0])
        class_name = dataset.classes[class_index]
        conf = box[5]
        
        cx, cy, w, h = box[1:5]
        cx *= width
        cy *= height
        w *= width
        h *= height
        rect = patches.Rectangle(
            (cx - (w/2), cy - (h/2)), 
            w, h, 
            linewidth = 2, edgecolor = 'r', facecolor = 'none'
        )
        ax.add_patch(rect)
        
        ax.text(
            cx - (w/2) - 10, cy - (h/2) - 10, 
            f"{class_name}: {conf:.2f}", 
            color = "white", fontsize = 10, backgroundcolor = "red"
        )
    plt.show()


#%%
from loader import YoloDataset
dataset = YoloDataset("train", 5, ["/home/pedro/Modelos/Faculdade/DL/Assignment 2/yolo_dataset"])
dataset.get_plot_box(4)

# %%
plot_images(dataset, 1000)

# %%
