# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def savetransformedimage(img_dir, image, name):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   transforms.ToPILImage()
                                   ])

    conv = invTrans(image['image'])
    conv.save(os.path.join(img_dir, 'ha', name))


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
        label = self.y[idx]

        image_res = np.array(image)
        image_res = self.transform(image=image_res)

        # savetransformedimage(self.img_dir, image_res, self.x.iloc[idx, 0])

        return image_res['image'], label


class CustomImageDatasetSingle(Dataset):
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

        image_res = np.array(image)
        image_res = self.transform(image=image_res)

        return image_res['image'], label
