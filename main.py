# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import os
from contextlib import redirect_stdout

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import csv

import Loader
import MyDataset
import Trainer
from DiseaseFlag import Diseases


def clean_then_save_csv(originalFilePath, cleanedFilePath, outputColumnName):
    with open(originalFilePath, newline='') as csvreadfile:
        with open(cleanedFilePath, 'w', newline='') as csvwritefile:
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
                    csv_write.writerow(['Image Index', outputColumnName])


if __name__ == "__main__":
    imageFolderPath = 'sample\\images\\'
    originalFilePath = 'sample\\sample_labels.csv'
    cleanedFilePath = 'sample\\sample_cleaned.csv'

    outputColumnName = 'Disease_numeric'

    clean_then_save_csv(originalFilePath, cleanedFilePath, outputColumnName)

    models = []
    for i in range(3):
        models.append(torchvision.models.alexnet(weights='DEFAULT'))

    # For the ones that use classifier layers
    print(models[0].classifier[-1])
    # For the ones that use fc as the last layer
    # print(models[0].fc)

    ImageNet_transforms = torchvision.models.AlexNet_Weights.DEFAULT.transforms()
    print(ImageNet_transforms)

    for model in models:
        # For the ones that use classifier layers
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=2, bias=True)
        # For the ones that use fc as the last layer
        # model..fc = nn.Linear(in_features=1024, out_features=2, bias=True)  # binary output

    transform = transforms.Compose(
        [
            transforms.Resize(256),  # Default is InterpolationMode.BILINEAR.
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    mySet = MyDataset.CustomImageDataset(cleanedFilePath, imageFolderPath, transform)
    loaders = Loader.create_loaders(mySet, train_ratio=0.8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_save_dir = 'content\\torch_logs'
    # tb_writer = SummaryWriter(model_save_dir)
    # tb_writer.add_graph(pretrained_torch_model.to("cpu"),
    #                     input_to_model=next(iter(loaders['train']))[0][0:])
    # tb_writer.close()
    #
    # Fine-tuning the ConvNet
    # trainer = Trainer.Trainer(pretrained_torch_model, "TorchPretrained", loaders,
    #                           0.001, device, tb_writer, False)

    train_length = 7

    log_name = 'save\\AlexNet_Adam_0.00001.log'
    state_name = 'save\\AlexNet_Adam_0.00001.pth'

    # Add momentum=0.9 for optimizer when using SGD for optimizer
    trainer1 = Trainer(models[0], "TorchPretrained", loaders, device=device,
                       logger=None, log=False, validation=True,
                       optimizer=torch.optim.Adam(models[0].parameters(), lr=0.00001),
                       loss=nn.CrossEntropyLoss(), epochs=train_length, logSave=log_name,
                       fileSave=state_name)

    log_name = 'save\\AlexNet_Adam_0.00002.log'
    state_name = 'save\\AlexNet_Adam_0.00002.pth'

    trainer2 = Trainer(models[1], "TorchPretrained", loaders, device=device,
                       logger=None, log=False, validation=True,
                       optimizer=torch.optim.Adam(models[1].parameters(), lr=0.00002),
                       loss=nn.CrossEntropyLoss(), epochs=train_length, logSave=log_name,
                       fileSave=state_name)

    log_name = 'save\\AlexNet_Adam_0.00005.log'
    state_name = 'save\\AlexNet_Adam_0.00005.pth'

    trainer3 = Trainer(models[2], "TorchPretrained", loaders, device=device,
                       logger=None, log=False, validation=True,
                       optimizer=torch.optim.Adam(models[2].parameters(), lr=0.00005),
                       loss=nn.CrossEntropyLoss(), epochs=train_length, logSave=log_name,
                       fileSave=state_name)

    trainer1.start()
    trainer2.start()
    trainer3.start()

    trainers = [trainer1, trainer2, trainer3]
    for t in trainers:
        t.join()
