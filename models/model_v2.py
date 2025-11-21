import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm


class RadiographyDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {'COVID-19': 0, 'Normal': 1,
                             'Viral Pneumonia': 2, 'Lung Opacity': 3}
        dirs = [data_dirs['COVID-19'], data_dirs['Normal'],
                data_dirs['Viral Pneumonia'], data_dirs['Lung Opacity']]
        classes = ['COVID-19', 'Normal', 'Viral Pneumonia', 'Lung Opacity']
        for folder, cls in zip(dirs, classes):
            print(f"Loading data from {cls}")
            for file in tqdm(os.listdir(folder)):
                self.images.append(os.path.join(folder, file))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
