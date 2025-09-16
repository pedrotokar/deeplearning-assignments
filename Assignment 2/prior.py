#%%
from sklearn.cluster import KMeans
import numpy as np
import os

from glob import glob

original_dataset_path = 'data/'
yolo_dataset_path = 'yolo_dataset'

files = glob(os.path.join(original_dataset_path, "labels", "train", "*.txt"))

annotations = list()
centers = list()

counter = 0
for txt_path in files:
    with open(txt_path) as f:
        for line in f:
            annotation = [float(x) for x in line.split()]
            annotations.append(annotation)

for annotation in annotations:
    centers.append([annotation[1], annotation[2]])

# print(centers)
# print(annotations)

#%%
n_anchors = 5
kmeans = KMeans(n_anchors)
kmeans.fit(centers)

print(kmeans.cluster_centers_)
