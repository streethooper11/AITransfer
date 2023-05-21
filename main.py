# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import csv

import Loader
import Models
import MyDataset
import Trainer
from DiseaseEnum import Disease


def training_stage(dataPath, imagePath, modelts, opts, savefolder,
                   outputNum=2, train_data_ratio=0.8, name=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []
    for modelf in modelts:
        for optim in opts:
            models.append(modelf(outputNum, device, optim))

    df = pd.read_csv(dataPath)

    if outputNum == 2:
        x = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=train_data_ratio,
                                                            stratify=y, random_state=0)

        trainDS = MyDataset.CustomImageDataset(imagePath, x_train, y_train)
        valDS = MyDataset.CustomImageDataset(imagePath, x_val, y_val)

        for model in models:
            pref = savefolder + model.name + model.opt[1] + model.opt[2] + '_decay_' + \
                   str(model.opt[3]) + '_' + str(train_data_ratio) + 'train'

            trainDS_copy = copy.deepcopy(trainDS)
            valDS_copy = copy.deepcopy(valDS)

            trainDS_copy.augmentation = model.augmentation
            trainDS_copy.transform = model.transform
            valDS_copy.transform = model.transform

            loader = Loader.create_loaders(train_set=trainDS_copy, val_set=valDS_copy)

            trainer = Trainer.Trainer(
                model.model,
                "TorchPretrained",
                loader,
                device=device,
                validation=True,
                optimizer=model.opt[0],
                loss=torch.nn.CrossEntropyLoss(),
                epochs=model.opt[4],
                pref=pref,
                name=name
            )

            trainer.train()

    else:
        x = df.iloc[:, 0:-outputNum]
        y = torch.tensor(df.iloc[:, -outputNum:].values)
        y = y.type(torch.float)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=train_data_ratio, random_state=0)

        trainDS = MyDataset.CustomImageDatasetMulti(imagePath, x_train, y_train)
        valDS = MyDataset.CustomImageDatasetMulti(imagePath, x_val, y_val)

        for model in models:
            suffix = model.name + model.opt[1] + model.opt[2] + '_decay_' + \
                   str(model.opt[3]) + '_trainratio_' + str(train_data_ratio)

            trainDS_copy = copy.deepcopy(trainDS)
            valDS_copy = copy.deepcopy(valDS)

            trainDS_copy.transform = model.transform
            valDS_copy.transform = model.transform

            loader = Loader.create_loaders(train_set=trainDS_copy, val_set=valDS_copy)

            trainer = Trainer.TrainerMulti(
                model.model,
                "TorchPretrained",
                loader,
                device=device,
                validation=True,
                optimizer=model.opt[0],
                loss=torch.nn.BCELoss(),
                epochs=model.opt[4],
                pref=savefolder,
                suffix=suffix,
                numoutputs=outputNum,
                listtops=3,
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


if __name__ == "__main__":
    modeltypes = [
        # Models.AlexNet,
        Models.MobileNetV2,
        # Models.EfficientNetB0,
        # Models.MobileNetV3L,
        # Models.RegNetY400MF,
        # Models.ResNet18,
        # Models.ResNet50,
        # Models.ResNext50,
        # Models.ShuffleNetV205,
        # Models.SqueezeNet10,
        # Models.Vgg11,
        # Models.VitB16,
    ]

    optims = [
#        (torch.optim.Adam, '_Adam_', '0.001', 0.0, 200),
        (torch.optim.Adam, '_Adam_', '0.0005', 0.0, 200),
    ]

    # variables you can change
    setnames = [
        'sample',
        'set3',
#        'set1-3',
    ]

    domultitogether = False  # True means multi-labels; false means work on each label separately
    column = Disease.HasDisease  # Used when multi-label flag is false; only work on this column
    torch.backends.cudnn.benchmark = True

    for setname in setnames:
        imageFolderPath = 'set\\' + setname + '\\train\\'

        if setname == 'sample':
            originalFilePath = 'set\\sample_labels.csv'
        else:
            originalFilePath = 'set\\Data_Entry_2017.csv'

        if domultitogether is True:
            sf = 'save\\' + setname + '\\multilabel\\'
            os.makedirs(sf, exist_ok=True)
            cleanedFilePath = sf + 'Entry_cleaned.csv'
            clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath,
                                colenum=-1)
            training_stage(cleanedFilePath, imageFolderPath, modeltypes, optims, sf,
                           outputNum=(len(Disease) - 1), train_data_ratio=0.8)
        else:
            sf = 'save\\' + setname + '\\' + column.name + '\\'
            os.makedirs(sf, exist_ok=True)
            cleanedFilePath = sf + 'Entry_cleaned.csv'
            clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath,
                                colenum=column.value)
            training_stage(cleanedFilePath, imageFolderPath, modeltypes, optims, sf,
                           outputNum=2, train_data_ratio=0.8, name=column.name)
