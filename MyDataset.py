# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms, InterpolationMode


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, istrain, train_ratio):
        self.img_dir = img_dir
        self.transform = None
        df = pd.read_csv(annotations_file)
        x = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, test_size=(1 - train_ratio),
                                                            stratify=y, random_state=0)

        if istrain:
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_test
            self.y = y_test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.x.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.y.iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
