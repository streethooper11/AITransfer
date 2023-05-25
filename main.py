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

import Loader
import Models
import MyDataset
import Trainer
from DiseaseEnum import Disease


def training_stage(dataPath, imagePath, modelts, opts, savefolder,
                   outputNum=2, train_ratio=0.8, name='', usesoftmax=False,
                   train_t=None, valid_t=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(dataPath)

    if usesoftmax is False:
        x = df.iloc[:, 0:-outputNum]
        y = torch.tensor(df.iloc[:, -outputNum:].values)
        y = y.type(torch.float)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=train_ratio, random_state=0)

        trainDS = MyDataset.CustomImageDataset(imagePath, x_train, y_train)
        valDS = MyDataset.CustomImageDataset(imagePath, x_val, y_val)

        for modelf in modelts:
            for optim in opts:
                model = modelf(outputNum, device, optim)

                trainDS_copy = copy.deepcopy(trainDS)
                valDS_copy = copy.deepcopy(valDS)

                trainDS_copy.transform = train_t
                valDS_copy.transform = valid_t

                loader = Loader.create_loaders(train_set=trainDS_copy, val_set=valDS_copy)

                trainer = Trainer.Trainer(
                    model,
                    loader,
                    device=device,
                    validation=True,
                    loss=torch.nn.BCELoss(),
                    pref=savefolder,
                    numoutputs=outputNum,
                    listtops=3,
                    name=name,
                    ratio=train_ratio
                )

                trainer.train()


    else:
        x = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=train_ratio,
                                                            stratify=y, random_state=11)

        trainDS = MyDataset.CustomImageDatasetSoftmax(imagePath, x_train, y_train)
        valDS = MyDataset.CustomImageDatasetSoftmax(imagePath, x_val, y_val)

        for modelf in modelts:
            for optim in opts:
                model = modelf(outputNum, device, optim)

                trainDS_copy = copy.deepcopy(trainDS)
                valDS_copy = copy.deepcopy(valDS)

                trainDS_copy.transform = model.augmentation
                valDS_copy.transform = model.transform

                loader = Loader.create_loaders(train_set=trainDS_copy, val_set=valDS_copy)

                trainer = Trainer.TrainerSoftmax(
                    model,
                    loader,
                    device=device,
                    validation=True,
                    loss=torch.nn.CrossEntropyLoss(),
                    savefolder=savefolder,
                    name=name,
                    ratio=train_ratio
                )

                trainer.train()


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


def doSet(modeltypelist, optlist, useset, domulti, resizedflag, usedcolumn, train_t, valid_t):
    if resized is True:
        imageFolderPath = os.path.join('set', useset, 'resized', '')
    else:
        imageFolderPath = os.path.join('set', useset, 'train', '')

    if useset == 'sample':
        originalFilePath = os.path.join('set', 'sample_labels.csv')
    else:
        originalFilePath = os.path.join('set', 'Data_Entry_2017.csv')

    if domulti is True:
        if resizedflag is True:
            sf = os.path.join('save', 'resized', useset, 'multilabel', '')
        else:
            sf = os.path.join('save', useset, 'multilabel', '')

        os.makedirs(sf, exist_ok=True)
        cleanedFilePath = os.path.join(sf, 'Entry_cleaned.csv')
        clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath,
                            colenum=-1)
        training_stage(cleanedFilePath, imageFolderPath, modeltypelist, optlist, sf,
                       outputNum=(len(Disease) - 1), train_ratio=0.7,
                       usesoftmax=False, train_t=train_t, valid_t=valid_t)
    else:
        if resizedflag is True:
            sf = os.path.join('save', 'resized', useset, usedcolumn.name, '')
        else:
            sf = os.path.join('save', useset, usedcolumn.name, '')

        os.makedirs(sf, exist_ok=True)
        cleanedFilePath = os.path.join(sf, 'Entry_cleaned.csv')
        clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath,
                            colenum=column.value)
        training_stage(cleanedFilePath, imageFolderPath, modeltypelist, optlist, sf,
                       outputNum=1, train_ratio=0.8, name=usedcolumn.name,
                       usesoftmax=False, train_t=train_t, valid_t=valid_t)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    modeltypes = [
        # Models.AlexNet,
        # Models.DenseNet121,
        # Models.MobileNetV2,
        # Models.EfficientNetB0,
        # Models.MobileNetV3L,
        # Models.RegNetY400MF,
        # Models.ResNet18,
        Models.ResNet50,
        # Models.ResNext50,
        # Models.ShuffleNetV205,
        # Models.SqueezeNet10,
        # Models.Vgg11,
        # Models.VitB16,
    ]

    optims = [
        (torch.optim.Adam, '_Adam_', '0.00001', 0.0, 50),
    ]

    # variables you can change
    setnames = [
        # 'sample',
        # 'set3',
        # 'set1-3',
        'set1-5',
        # 'all',
    ]

    domultilabels = False  # True means multi-labels; false means work on each label separately
    resized = False  # True means use previously resized images
    column = Disease.HasDisease  # Used when multi-label flag is false; only work on this column

    translate_per = dict()
    translate_per['x'] = (-0.1, 0.1)
    translate_per['y'] = (-0.2, 0)

    train_transform = A.Compose(
        [
            A.CLAHE(p=1.0),
            A.GaussNoise(per_channel=False, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Affine(scale=None, translate_percent=translate_per, rotate=(-10, 10), always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Resize(224, 224),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            A.CLAHE(p=1.0),
            A.Resize(224, 224),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    for setname in setnames:
        doSet(modeltypes, optims, setname, domultilabels, resized, column, train_transform, valid_transform)

    resized = True
    train_transform = A.Compose(
        [
            A.CLAHE(p=1.0),
            A.GaussNoise(per_channel=False, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Affine(scale=None, translate_percent=translate_per, rotate=(-10, 10), always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            # A.Resize(224, 224),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            A.CLAHE(p=1.0),
            # A.Resize(224, 224),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    for setname in setnames:
        doSet(modeltypes, optims, setname, domultilabels, resized, column, train_transform, valid_transform)
