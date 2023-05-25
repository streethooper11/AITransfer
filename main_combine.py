# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os
import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import csv
import albumentations as A
import albumentations.pytorch
from albumentations.augmentations import transforms

import Loader
import Models
import MyDataset
import Trainer_combine
from DiseaseEnum import Disease


def training_stage(dataPath, imagePath, best_path, model_t, optim_t, savefolder,
                   outputNum=1, train_ratio=0.8, name='',
                   train_t=None, valid_t=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(dataPath)

    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=train_ratio, random_state=11)

    trainDS = MyDataset.CustomImageDatasetSingle(imagePath, x_train, y_train)
    trainDS.transform = train_t

    valDS = MyDataset.CustomImageDatasetSingle(imagePath, x_val, y_val)
    valDS.transform = valid_t

    loader = Loader.create_loaders(train_set=trainDS, val_set=valDS)

    model = model_t(outputNum, device, optim_t)
    if best_path is not None:
        checkpoint = torch.load(best_path)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.opt[0].load_state_dict(checkpoint['optimizer_state_dict'])

    trainer = Trainer_combine.TrainerCombineSingle(
        model,
        loader,
        device=device,
        validation=True,
        loss=torch.nn.BCELoss(),
        pref=savefolder,
        name=name,
        ratio=train_ratio
    )

    best_model = trainer.train()

    return best_model

def clean_then_save_csv(origFilePath, cleanFilePath, imgPath, colenum):
    with open(origFilePath, newline='') as csvreadfile:
        with open(cleanFilePath, 'w', newline='') as csvwritefile:
            csv_read = csv.reader(csvreadfile, delimiter=',')
            csv_write = csv.writer(csvwritefile, delimiter=',')
            firstrow = True

            if colenum == -1:
                for row in csv_read:
                    if firstrow is False:
                        img_path = os.path.join(imgPath, row[0])
                        if os.path.exists(img_path):
                            diseases = [0 for n in range(len(Disease) - 1)]
                            if row[1] == "No Finding":
                                values = [row[0]]
                                values.extend(diseases)
                                csv_write.writerow(values)
                            else:
                                diseasefindings = row[1].split('|')
                                for eachdisease in diseasefindings:
                                    diseases[Disease[eachdisease].value] = 1
                                values = [row[0]]
                                values.extend(diseases)
                                csv_write.writerow(values)
                    else:
                        firstrow = False
                        csv_write.writerow(['Image Index'] +
                                           [d.name for d in Disease if d.value < 20])

            elif colenum == Disease.HasDisease.value:
                for row in csv_read:
                    if firstrow is False:
                        img_path = os.path.join(imgPath, row[0])
                        #if row[6] == 'PA':
                        if os.path.exists(img_path):
                            if row[1] == "No Finding":
                                csv_write.writerow([row[0], 0])
                            else:
                                csv_write.writerow([row[0], 1])

                    else:
                        firstrow = False
                        csv_write.writerow(['Image Index', Disease.HasDisease.name])

            else:
                diseasecol = ''
                for data in Disease:
                    if data.value == colenum:
                        diseasecol = data.name
                        break

                for row in csv_read:
                    if firstrow is False:
                        img_path = os.path.join(imgPath, row[0])
                        if os.path.exists(img_path):
                            diseasefindings = row[1].split('|')
                            if diseasecol in diseasefindings:
                                csv_write.writerow([row[0], 1])
                            else:
                                csv_write.writerow([row[0], 0])
                    else:
                        firstrow = False
                        csv_write.writerow(['Image Index', diseasecol])


def doSet(best_path, model_t, optim_t, useset, resizedflag, usedcolumn, train_t, valid_t):
    if resized is True:
        imageFolderPath = os.path.join('set', 'all', 'resized', useset, '')
    else:
        imageFolderPath = os.path.join('set', 'all', useset, '')

    if useset == 'sample':
        originalFilePath = os.path.join('set', 'sample_labels.csv')
    else:
        originalFilePath = os.path.join('set', 'Data_Entry_2017.csv')

    if resizedflag is True:
        sf = os.path.join('save', 'resized', 'combine', usedcolumn.name, useset, '')
    else:
        sf = os.path.join('save', 'combine', usedcolumn.name, useset, '')

    os.makedirs(sf, exist_ok=True)
    cleanedFilePath = os.path.join(sf, 'Entry_cleaned.csv')
    clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath,
                        colenum=column.value)
    best_model = training_stage(cleanedFilePath, imageFolderPath, best_path, model_t, optim_t, sf,
                                outputNum=1, train_ratio=0.8, name=usedcolumn.name,
                                train_t=train_t, valid_t=valid_t)

    return best_model

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # variables you can change
    setnames = [
        # 'set1',
        'set2',
        'set3',
        'set4',
        'set5',
        'set6',
        'set7',
        'set8',
        'set9',
        'set10',
        'set11',
        # 'set12',
    ]

    resized = True  # True means use previously resized images
    column = Disease.HasDisease  # Used when multi-label flag is false; only work on this column

    translate_per = dict()
    translate_per['x'] = (-0.1, 0.1)
    translate_per['y'] = (-0.1, 0)

    train_transform = A.Compose(
        [
            # A.CLAHE(always_apply=True, p=1.0),
            # A.GaussNoise(per_channel=False, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, always_apply=True, p=1.0),
            A.Affine(scale=None, translate_percent=translate_per, rotate=(-10, 10), always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            # A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            # A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            # A.CLAHE(always_apply=True, p=1.0),
            # A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            # A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    modeltype = Models.MobileNetV2
    optim = (torch.optim.Adam, '_Adam_', '0.0001', 0.0, 50)

    bestmodelpath = None
    for setname in setnames:
        bestmodelpath = doSet(bestmodelpath, modeltype, optim, setname, resized,
                              column, train_transform, valid_transform)

    resized = False

    train_transform = A.Compose(
        [
            # A.CLAHE(always_apply=True, p=1.0),
            # A.GaussNoise(per_channel=False, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, always_apply=True, p=1.0),
            A.Affine(scale=None, translate_percent=translate_per, rotate=(-10, 10), always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Resize(224, 224),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            # A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            # A.CLAHE(always_apply=True, p=1.0),
            A.Resize(224, 224),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            # A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    bestmodelpath = None
    for setname in setnames:
        bestmodelpath = doSet(bestmodelpath, modeltype, optim, setname, resized,
                              column, train_transform, valid_transform)
