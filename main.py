# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import csv

import Loader
import Models
import MyDataset
import Trainer
from DiseaseEnum import Disease


def training_stage(dataPath, imagePath, modelts, opts, savefolder,
                   outputNum=2, train_data_ratio=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    models = []
    for modelf in modelts:
        for optim in opts:
            models.append(modelf(outputNum, device, optim))

    if outputNum == 2:
        trainDS = MyDataset.CustomImageDataset(dataPath, imagePath, True, train_data_ratio)
        testDS = MyDataset.CustomImageDataset(dataPath, imagePath, False, train_data_ratio)

        for model in models:
            pref = savefolder + model.name + model.opt[1] + model.opt[2] + '_decay_' + \
                   str(model.opt[3]) + '_' + str(train_data_ratio) + 'train'

            trainDS_copy = copy.deepcopy(trainDS)
            testDS_copy = copy.deepcopy(testDS)

            trainDS_copy.transform = model.transform
            testDS_copy.transform = model.transform

            loader = Loader.create_loaders(trainDS_copy, testDS_copy)

            trainer = Trainer.Trainer(
                model.model,
                "TorchPretrained",
                loader,
                device=device,
                validation=True,
                optimizer=model.opt[0],
                loss=torch.nn.CrossEntropyLoss(),
                epochs=model.opt[4],
                logSave=pref + '.log',
                fileSave=pref + '.pth',
            )

            trainer.train()

    else:
        trainDS = MyDataset.CustomImageDatasetMulti(dataPath, imagePath, True, train_data_ratio)
        testDS = MyDataset.CustomImageDatasetMulti(dataPath, imagePath, False, train_data_ratio)

        for model in models:
            pref = savefolder + model.name + model.opt[1] + model.opt[2] + '_decay_' + \
                   str(model.opt[3]) + '_' + str(train_data_ratio) + 'train'

            trainDS_copy = copy.deepcopy(trainDS)
            testDS_copy = copy.deepcopy(testDS)

            trainDS_copy.transform = model.transform
            testDS_copy.transform = model.transform

            loader = Loader.create_loaders(trainDS_copy, testDS_copy)

            trainer = Trainer.TrainerMulti(
                model.model,
                "TorchPretrained",
                loader,
                device=device,
                validation=True,
                optimizer=model.opt[0],
                loss=torch.nn.MultiLabelMarginLoss(),
                epochs=model.opt[4],
                logSave=pref + '.log',
                fileSave=pref + '.pth',
                numoutputs=outputNum
            )

            trainer.train()


def clean_then_save_csv(origFilePath, cleanFilePath, imgPath, colenum=-1):
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
#        (torch.optim.Adam, '_Adam_', '0.00001', 0.0, 25),
#        (torch.optim.Adam, '_Adam_', '0.00003', 0.0, 25),
#        (torch.optim.Adam, '_Adam_', '0.00005', 0.0, 25),
        (torch.optim.Adam, '_Adam_', '0.0001', 0.0, 25),
        (torch.optim.Adam, '_Adam_', '0.0003', 0.0, 25),
        (torch.optim.Adam, '_Adam_', '0.0005', 0.0, 25),
        (torch.optim.Adam, '_Adam_', '0.001', 0.0, 25),
    ]

    output = len(Disease) - 1
    # output = 2

    imageFolderPath = 'set\\set12\\images\\'
    originalFilePath = 'set\\Data_Entry_2017.csv'
    sf = 'save\\multiclass\\set12\\'
    os.makedirs(sf, exist_ok=True)
    cleanedFilePath = sf + 'Entry_cleaned.csv'

    clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath,
                        colenum=-1)
    training_stage(cleanedFilePath, imageFolderPath, modeltypes, optims, sf,
                   outputNum=output)
