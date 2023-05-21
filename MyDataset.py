# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms


class CustomImageDatasetMulti(Dataset):
    def __init__(self, img_dir, x, y):
        self.img_dir = img_dir
        self.augmentation = None
        self.transform = None
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.x.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, x, y):
        self.img_dir = img_dir
        self.augmentation = None
        self.transform = None
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.x.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.y.iloc[idx]

        if self.augmentation:
            image = self.augmentation(image)

        image = self.transform(image)

        red_mean = torch.mean(image[0, :, :])  # we take mean pixel value from Red channel
        green_mean = torch.mean(image[1, :, :])  # we take mean pixel value from Green channel
        blue_mean = torch.mean(image[2, :, :])  # we take mean pixel value from Blue channel

        red_std = torch.std(image[0, :, :])  # we take std deviation of pixel values from Red channel
        green_std = torch.std(image[1, :, :])  # we take std deviation of pixel values from Green channel
        blue_std = torch.std(image[2, :, :])  # we take std deviation of pixel values from Blue channel

        norm_means = [red_mean, green_mean, blue_mean]
        norm_stds = [red_std, green_std, blue_std]

        transform = transforms.Normalize(mean=norm_means, std=norm_stds)
        image = transform(image)

        return image, label
