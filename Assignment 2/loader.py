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

    def __init__(self, type_, dataset_path = ["yolo_dataset"], device = "cpu"):
        self.base_path_data = os.path.join(*dataset_path, "images", f"{type_}")
        self.base_path_labels = os.path.join(*dataset_path, "labels", f"{type_}")

        self.images = glob(os.path.join(self.base_path_data, "*.jpg"))

        self.device = device

        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.images)
    
    #Feito na inicialização da classe para pegar as prioris de caixas
    def clusterize_anchors(self, n_anchors):
        self.n_anchors = n_anchors

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
            centers.append([annotation[3], annotation[4]])
        
        kmeans = KMeans(self.n_anchors)
        kmeans.fit(centers)

        self.anchors = torch.tensor(kmeans.cluster_centers_)
    
    def set_anchors(self, anchors):
        self.anchors = anchors
        self.n_anchors = len(anchors)

    def get_anchors(self):
        return self.anchors

    def letterbox_data(self, image, boxes, new_dim):
        if image.shape[2] > image.shape[1]: #W > H
            scale_factor = new_dim/image.shape[2]
            new_H = int(scale_factor * image.shape[1])
            new_dims = (new_H, new_dim)

            resize = T.Resize(new_dims)
            padding = T.Pad([0, (new_dim - new_H)//2], fill = 117)

            box_scale = new_dims[0]/new_dim
            for box in boxes:
                box[2] = ((box[2] - 0.5) * box_scale) + 0.5
                box[4] *= box_scale


        elif image.shape[2] < image.shape[1]: #W < H
            scale_factor = new_dim/image.shape[1]
            new_W = int(scale_factor * image.shape[2])
            new_dims = (new_dim, new_W)

            resize = T.Resize(new_dims)
            padding = T.Pad([(new_dim - new_W)//2, 0], fill = 117)
            box_scale = new_dims[1]/new_dim
            for box in boxes:
                box[1] = ((box[1] - 0.5) * box_scale) + 0.5
                box[3] *= box_scale

        else: #W = H
            resize = T.Resize((new_dim, new_dim))
            padding = lambda x: x
            mode = "E"
        
        return padding(resize(image)), boxes

    def iou_wh(self, box_1, box_2):
        intersection = torch.min(box_1[0], box_2[:, 0]) * torch.min(box_1[1], box_2[:, 1])
        union = (box_1[0] * box_1[1]) + (box_2[:, 0] * box_2[:, 1]) - intersection
        return intersection / union

    # TODO: otimizar e modularizar
    def __getitem__(self, idx):
        dim = 416
        #Pega o path da imagem e da label e carrega
        img_path = self.images[idx]
        name = img_path.split("/")[-1].split(".")[0]
        label_path = os.path.join(self.base_path_labels, f"{name}.txt")

        #Carrega as boxes e muda de acordo com o letterbox (talvez separar em outra função? n sei)
        boxes = []
        with open(label_path, "r") as f:
            for box in f:
                boxes.append([float(x) for x in box.split()])
        
        #Dá letterbox na imagem e nas boxes
        image = decode_image(img_path).to(torch.float).to(self.device)
        letterbox_image, boxes = self.letterbox_data(image, boxes, dim)

        #Agora com a imagem e boxes no tamanho certo, cria as boxes no formato do modelo
        strides = [8, 16, 32] #Pro modelo vai ser 8 16 32 mas deixei 32 pq não vi as outras saidas do modelo...
        matrixes = [] #Vai ter as saídas pra cada stride
        for stride in strides:
            if dim % stride != 0:
                raise ValueError(f"Dimensão {dim} incompatível com o stride {stride}")
            n_cells = dim//stride
            cell_relative_size = 1/n_cells

            output_matrix = torch.zeros((self.n_anchors, n_cells, n_cells, 5 + self.n_classes))

            for box in boxes:
                cell_x, cell_y = int(box[1]/cell_relative_size), int(box[2]/cell_relative_size) #H depois W

                dx = box[1] * n_cells - cell_x
                dy = box[2] * n_cells - cell_y

                IoUs = self.iou_wh(torch.tensor(box[3:5]), self.anchors)
                assigned_anchor_index = IoUs.argmax()
                if output_matrix[assigned_anchor_index, cell_y, cell_x, 0] == 1:
                    raise Exception("Duas bounding boxes anotadas ficaram associadas à mesma célula e a mesma anchor")
                #Meu amigo gem tá dizendo que é melhor passar o log aqui do que passar a exponencial na função
                pw = torch.log(box[3] / self.anchors[assigned_anchor_index][0])
                ph = torch.log(box[4] / self.anchors[assigned_anchor_index][1])

                #Talvez esse retorno de matriz seja questionável e algo como um dicionário faça mais sentido.
                output_matrix[assigned_anchor_index, cell_y, cell_x, :5] = torch.tensor([1, dx, dy, pw, ph])
                output_matrix[assigned_anchor_index, cell_y, cell_x, 5 + int(box[0])] = 1
            matrixes.append(output_matrix)

        return letterbox_image, matrixes
    
    #Função auxiliar para pegar as boxes em um formato adequado para plotagem
    def get_plot_box(self, idx):
        img_path = self.images[idx]
        name = img_path.split("/")[-1].split(".")[0]
        label_path = os.path.join(self.base_path_labels, f"{name}.txt")

        boxes = []
        with open(label_path, "r") as f:
            for box in f:
                boxes.append([*[float(x) for x in box.split()], 1]) #adiciona confiança

        return boxes

    #Função auxiliar para pegar a imagem original sem normalização
    def get_plot_image(self, idx):
        img_path = self.images[idx]
        return decode_image(img_path)

#%%
dataset = YoloDataset("train", 5, ["/home/pedro/Modelos/Faculdade/DL/Assignment 2/yolo_dataset"])
dataset.get_plot_box(4)

# %%
