# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import subprocess

import torch
import torchvision
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import csv
from multiprocessing import JoinableQueue

import Loader
import Models
import MyDataset
import Trainer
from DiseaseFlag import Diseases


def clean_then_save_csv(origFilePath, cleanFilePath, outputColName):
    with open(origFilePath, newline='') as csvreadfile:
        with open(cleanFilePath, 'w', newline='') as csvwritefile:
            csv_read = csv.reader(csvreadfile, delimiter=',')
            csv_write = csv.writer(csvwritefile, delimiter=',')
            firstrow = True
            for row in csv_read:
                if firstrow is False:
                    if row[1] == "No Finding":
                        csv_write.writerow([row[0], 0])
                    else:
                        csv_write.writerow([row[0], 1])
                else:
                    firstrow = False
                    csv_write.writerow(['Image Index', outputColName])


if __name__ == "__main__":
    imageFolderPath = 'sample\\images\\'
    originalFilePath = 'sample\\sample_labels.csv'
    sf = 'save\\'
    cleanedFilePath = 'sample\\sample_cleaned.csv'

    outputColumnName = 'Disease_numeric'

    clean_then_save_csv(originalFilePath, cleanedFilePath, outputColumnName)

    # model = torchvision.models.efficientnet_b0(weights='DEFAULT')
    # For the ones that use classifier layers
    # print(model.classifier[-1])
    # For the ones that use fc as the last layer
    # print(model.fc)
    # ImageNet_transforms = torchvision.models.MaxVit_T_Weights.DEFAULT.transforms()
    # print(ImageNet_transforms)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    lr = 0.000002
    optim = torch.optim.Adamax
    ot = '_Adamax_'

    outputNum = 2

    models = [
        Models.AlexNet(outputNum),
        Models.EfficientNetB0(outputNum),
        Models.MobileNetV3L(outputNum),
        Models.RegNetY400MF(outputNum),
        Models.ResNet18(outputNum),
        Models.ResNet50(outputNum),
        Models.ResNext50(outputNum),
        Models.ShuffleNetV205(outputNum),
        Models.SqueezeNet10(outputNum),
        Models.VitB16(outputNum),
        Models.Vgg11(outputNum),
    ]

    max_procs = 2
    queue = JoinableQueue(max_procs)
    train_length = 8
    train_ratio = 0.8

    myDS = MyDataset.CustomImageDataset(cleanedFilePath, imageFolderPath)
    torch.manual_seed(0)
    train_test_set = random_split(myDS, [train_ratio, (1 - train_ratio)])

    for model in models:
        train_test_set[0].dataset.transform = model.transform
        train_test_set[1].dataset.transform = model.transform

        for optimtuple in model.optimwithstr:
            queue.put(0)

            pref = sf + model.name + optimtuple[1] + optimtuple[2]
            train_test_copy = copy.deepcopy(train_test_set)

            loader = Loader.create_loaders(train_test_copy)

            Trainer.Trainer(model.model, "TorchPretrained", loader, device=device,
                            validation=True,
                            optimizer=optimtuple[0],
                            loss=nn.CrossEntropyLoss(), epochs=train_length,
                            logSave=pref + '.log',
                            fileSave=pref + '.pth',
                            jobs=queue).start()

    queue.join()
