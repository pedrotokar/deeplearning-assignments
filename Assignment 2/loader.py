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
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from tqdm import tqdm

import os
from glob import glob


#%%
class YoloDataset(Dataset):

    def __init__(self, type_, n_anchors, dataset_path = ["yolo_dataset"], device = "cpu"):
        self.base_path_data = os.path.join(*dataset_path, "images", f"{type_}")
        self.base_path_labels = os.path.join(*dataset_path, "labels", f"{type_}")

        self.images = glob(os.path.join(self.base_path_data, "*.jpg"))

        self.device = device

        self.n_anchors = n_anchors
        self.clusterize_boxes()
        print(self.anchors)
    
    def __len__(self):
        return len(self.images)
    
    #Feito na inicialização da classe para pegar as prioris de caixas
    def clusterize_boxes(self):
        labels = glob(os.path.join(self.base_path_labels, "*.txt"))

        annotations = list()
        centers = list()

        counter = 0
        for txt_path in labels:
            with open(txt_path, "r") as f:
                for line in f:
                    annotation = [float(x) for x in line.split()]
                    annotations.append(annotation)

        for annotation in annotations:
            centers.append([annotation[1], annotation[2]])
        
        kmeans = KMeans(self.n_anchors)
        kmeans.fit(centers)

        self.anchors = kmeans.cluster_centers_

    def letterbox_image(self, image, desired_dims):
        if image.shape[2] > image.shape[1]: #W > H
            scale_factor = desired_dims/image.shape[2]
            new_H = int(scale_factor * image.shape[1])
            new_dims = (new_H, desired_dims)

            resize = T.Resize(new_dims)
            padding = T.Pad([0, (desired_dims - new_H)//2], fill = 117)
            mode = "W"

        elif image.shape[2] < image.shape[1]: #W < H
            scale_factor = desired_dims/image.shape[1]
            new_W = int(scale_factor * image.shape[2])
            new_dims = (desired_dims, new_W)

            resize = T.Resize(new_dims)
            padding = T.Pad([(desired_dims - new_W)//2, 0], fill = 117)
            mode = "H"

        else: #W = H
            resize = T.Resize((desired_dims, desired_dims))
            padding = lambda x: x
            mode = "E"
        
        return padding(resize(image)), new_dims, mode

    # TODO: terminar
    def __getitem__(self, idx):
        desired_dim = 416
        #Pega o path da imagem e da label e carrega
        img_path = self.images[idx]
        name = img_path.split("/")[-1].split(".")[0]
        label_path = os.path.join(self.base_path_labels, f"{name}.txt")

        image = decode_image(img_path).to(torch.float).to(self.device)
        letterbox_image, new_dims, mode = self.letterbox_image(image, desired_dim)

        boxes = []
        with open(label_path, "r") as f:
            for box in f:
                boxes.append([float(x) for x in box.split()])
        
        if mode == "W":
            box_scale = new_dims[0]/desired_dim
            c_y_offset = (1 - box_scale)/2
            for box in boxes:
                box[2] += c_y_offset
                box[4] *= box_scale
        elif mode == "H":
            box_scale = new_dims[1]/desired_dim
            c_x_offset = (1 - box_scale)/2
            for box in boxes:
                box[1] += c_x_offset
                box[3] *= box_scale

        #O que falta:
        #Calcular com o stride as dimensões da matriz de retorno
        #Aplicar técnicas para associar à bounding box certa

        return letterbox_image, boxes
    
    #Função auxiliar para pegar as boxes em um formato adequado para plotagem
    def get_plot_box(self, idx):
        img_path = self.images[idx]
        name = img_path.split("/")[-1].split(".")[0]
        label_path = os.path.join(self.base_path_labels, f"{name}.txt")

        boxes = []
        with open(label_path, "r") as f:
            for box in f:
                boxes.append([*[float(x) for x in box.split()], 1])

        return boxes

    #Função auxiliar para pegar a imagem original sem normalização
    def get_plot_image(self, idx):
        img_path = self.images[idx]
        return decode_image(img_path)


#%%
dataset = YoloDataset("train", 5, ["/home/pedro/Modelos/Faculdade/DL/Assignment 2/yolo_dataset"])
dataset.get_plot_box(4)

# %%
