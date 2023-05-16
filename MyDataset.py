# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import os
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDatasetMulti(Dataset):
    def __init__(self, img_dir, x, y):
        self.img_dir = img_dir
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
        self.transform = None
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.x.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.y.iloc[idx]

        if self.transform:
            image = self.transform(image)

        self.x.iloc[idx, 0] = image
        inputs = self.x.iloc[idx]

        return inputs, label
