# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH
import copy

import torch
import torchvision
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import csv
import queue

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sf = 'save\\'
    lr = 0.00001
    optim = torch.optim.Adamax
    ot = '_Adamax_'

    outputNum = 2
    models = [
        Models.AlexNet(optim, ot, outputNum),
        Models.EfficientNetB0(optim, ot, outputNum),
        Models.MaxVitT(optim, ot, outputNum),
        Models.MNasNet05(optim, ot, outputNum),
        Models.MobileNetV3L(optim, ot, outputNum),
        Models.RegNetY400MF(optim, ot, outputNum),
        Models.ResNet18(optim, ot, outputNum),
        Models.ResNet50(optim, ot, outputNum),
        Models.ResNext50(optim, ot, outputNum),
        Models.ShuffleNetV205(optim, ot, outputNum),
        Models.SqueezeNet10(optim, ot, outputNum),
        Models.VitB16(optim, ot, outputNum),
        Models.Vgg11(optim, ot, outputNum)
    ]

    max_threads = 2
    queue = queue.Queue(max_threads)
    train_length = 15
    train_ratio = 0.8

    myDS = MyDataset.CustomImageDataset(cleanedFilePath, imageFolderPath)
    torch.manual_seed(0)
    train_test_set = random_split(myDS, [train_ratio, (1 - train_ratio)])

    for model in models:
        queue.put(0)

        pref = sf + model.name + model.optstr + str(lr)
        train_test_copy = copy.deepcopy(train_test_set)
        train_test_copy[0].dataset.transform = model.transform
        train_test_copy[1].dataset.transform = model.transform

        loader = Loader.create_loaders(train_test_copy)

        Trainer.Trainer(model.model, "TorchPretrained", loader, device=device,
                        logger=None, log=False, validation=True,
                        optimizer=optim(model.model.parameters(), lr=lr),
                        loss=nn.CrossEntropyLoss(), epochs=train_length,
                        logSave=pref + '.log',
                        fileSave=pref + '.pth',
                        jobs=queue).start()

    queue.join()
