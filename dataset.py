import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transform import *
import cv2
import os 
import config
from PIL import Image
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


class Mydataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------
        self.images = sorted(os.listdir(str(image_dir)))
        self.labels = sorted(os.listdir(str(label_dir)))
        assert len(self.images) == len(self.labels)
        self.transform =transform 
        
        self.images_and_label = []
        for i in range(len(self.images)):
            self.images_and_label.append((str(image_dir)+'/'+str(self.images[i]),
                                          str(label_dir)+'/'+str(self.labels[i])))

    def __getitem__(self, index):
        
        image_path, label_path = self.images_and_label[index]
        
        image = Image.open(image_path).convert("L") # PIL: RGB, OpenCV: BGR
        cv2.imwrite("input1.png", np.array(image))
        
        label = Image.open(label_path).convert("L")       
        cv2.imwrite("label1.png", np.array(label))
        
        if self.transform is not None:
            image, label = self.transform((image,label))
        return image, label
    
    def __len__(self):
        return len(self.images)
