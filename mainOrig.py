# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy

import torch
from torch import nn, optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import csv
from multiprocessing import JoinableQueue

import Loader
import Models
import MyDatasetOrig
import Trainer
from DiseaseEnum import Diseases


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

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    modeltypes = [
        Models.AlexNet,
        # Models.EfficientNetB0,
        # Models.MobileNetV3L,
        # Models.RegNetY400MF,
        Models.ResNet18,
        # Models.ResNet50,
        # Models.ResNext50,
        # Models.ShuffleNetV205,
        # Models.SqueezeNet10,
        Models.Vgg11,
        # Models.VitB16,
    ]

    optims = [
        (optim.Adam, '_Adam_', '0.0001'),
        (optim.Adagrad, '_Adagrad_', '0.0001'),
        (optim.Adamax, '_Adamax_', '0.0001'),
        (optim.RMSprop, '_RMSProp_', '0.0001'),
        (optim.SGD, '_SGD_', '0.0001'),
    ]

    outputNum = 2

    models = []
    for modelf in modeltypes:
        for optim in optims:
            models.append(modelf(outputNum, device, optim))

    max_procs = 1
    queue = JoinableQueue(max_procs)
    train_length = 6
    train_ratio = 0.8

    myDS = MyDataset.CustomImageDataset(cleanedFilePath, imageFolderPath)
    torch.manual_seed(0)
    train_test_set = random_split(myDS, [train_ratio, (1 - train_ratio)])

    for model in models:
        queue.put(0)

        pref = sf + model.name + model.opt[1] + model.opt[2]

        train_test_copy = copy.deepcopy(train_test_set)
        train_test_copy[0].dataset.transform = model.transform
        train_test_copy[1].dataset.transform = model.transform

        loader = Loader.create_loaders(train_test_copy)

        Trainer.Trainer(
            model.model,
            "TorchPretrained",
            loader,
            device=device,
            validation=True,
            optimizer=model.opt[0],
            loss=nn.CrossEntropyLoss(), epochs=train_length,
            logSave=pref + '.log',
            fileSave=pref + '.pth',
            jobs=queue
        ).start()

    queue.join()
