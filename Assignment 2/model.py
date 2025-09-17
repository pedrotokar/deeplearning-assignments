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
from PIL import Image

from tqdm import tqdm

import os
import glob

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"using device '{device}'")

class ModelTraining:
    def __init__():
        self.history = {
            "epochs": 0,
            "train": {
                "loss": [],
#                "accuracy": []
            },
            "validation":{
                "loss": [],
#                "accuracy": []
            }
        }

    def save_model(self, path = None):
        if path:
            json_path = os.path.join(*path, f"{self.__name__}.json")    
            pth_path = os.path.join(*path, f"{self.__name__}.pth")
        else:
            json_path = f"{self.__name__}.json"
            pth_path = f"{self.__name__}.pth"
        with open(json_path, "w", encoding = "utf-8") as f:
            json.dump(self.history, f, ensure_ascii = False, indent = 4)
        torch.save(self.state_dict(), pth_path)
    
    def load_model(self, path = None):
        if path:
            json_path = os.path.join(*path, f"{self.__name__}.json")    
            pth_path = os.path.join(*path, f"{self.__name__}.pth")
        else:
            json_path = f"{self.__name__}.json"
            pth_path = f"{self.__name__}.pth"
        with open(json_path, "r", encoding = "utf-8") as f:
            self.history = json.load(f)
        self.load_state_dict(torch.load(pth_path, weights_only = True))

    def plot_hist(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Histórico de treinamento - {self.__name__}")

        axes[0].plot(hist["train"]["loss"], label = "Train")
        axes[0].plot(hist["val"]["loss"], label = "Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss de treino")
    #    axes[0].set_title("Comparação de perda no treino")
        axes[0].legend()
    #     axes[1].plot(hist["train"]["acc"], label = "Train")
    #     axes[1].plot(hist["val"]["acc"], label = "Validation")
    #     axes[1].set_xlabel("Epoch")
    #     axes[1].set_ylabel("Acurácia")
    #     axes[1].set_ylim((0, 1))
    #     axes[1].set_title("Comparação de acurácia no treino")
    #     axes[1].legend()

    def fit(self, train_loader, val_loader, loss_fc, optimizer, epochs = 40):
        print("Épocas treinadas anteriormente: {self.history['epochs']}")
        
        for epoch in range(self.history["epochs"], self.history["epochs"] + epochs):  
            
            progress_bar = tqdm(train_loader, desc = f"Epoch {epoch + 1} | train")
            running_loss_train = 0

            self.train()
            for X, y in progress_bar:
                pred = self(X)
                loss = loss_fc(pred, y)

                optimizer.zero_grad()
                loss.backward()
                otimizador.step()

                running_loss_train += loss.item() * X.size(0)
                # _, predicted_labels = torch.max(pred, 1)
                # correct_predictions_train += (predicted_labels == y).sum().item()
                # total_samples += torch.numel(y)

                progress_bar.set_postfix(loss = loss.item())
            
            progress_bar = tqdm(val_loader, desc = f"Epoch {epoch + 1} | train")
            running_loss_val = 0

            self.eval()
            for X, y in progress_bar:
                pred = self(X)
                loss = loss_fc(pred, y)

                running_loss_val += loss.item() * X.size(0)
                
                progress_bar.set_postfix(loss = loss.item())
            
            scheduler.step(epoch_loss_val)
            print(f"Fim Epoch {epoch+1}:")
            print(f"   -> Loss train: {epoch_loss_train:.6f} | Loss val: {epoch_loss_val:.6f}")
#            print(f"   -> Acc train: {epoch_acc_train:.6f}  | Acc val: {epoch_acc_val:.6f}")
            print(f"   -> LR: {scheduler.get_last_lr()[0]:.6f}\n")

            self.history["train"]["loss"].append(epoch_loss_train)
            self.history["validation"]["loss"].append(epoch_loss_val)
            self.history["epochs"] = epoch

            if epoch % 5 == 0:
                self.save_model() #Pro colab não explodir o modelo igual já aconteceu ;-;
