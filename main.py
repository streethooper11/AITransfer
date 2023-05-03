# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

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

    outputNum = 2

    clean_then_save_csv(originalFilePath, cleanedFilePath, outputColumnName)

    # model = torchvision.models.efficientnet_b0(weights='DEFAULT')
    # For the ones that use classifier layers
    # print(model.classifier[-1])
    # For the ones that use fc as the last layer
    # print(model.fc)
    ImageNet_transforms = torchvision.models.MaxVit_T_Weights.DEFAULT.transforms()
    print(ImageNet_transforms)

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

    models = [
        Models.AlexNet(outputNum),
        Models.EfficientNetB0(outputNum),
        Models.MaxVitT(outputNum),
        Models.MNasNet05(outputNum),
        Models.MobileNetV3L(outputNum),
        Models.RegNetY400MF(outputNum),
        Models.ResNet18(outputNum),
        Models.ResNet50(outputNum),
        Models.ResNext50(outputNum),
        Models.ShuffleNetV205(outputNum),
        Models.SqueezeNet10(outputNum),
        Models.VitB16(outputNum),
        Models.Vgg11(outputNum)
    ]

    queue = queue.Queue(3)  # Max 3 threads

    train_length = 15
    train_ratio = 0.8

    sf = 'save\\'
    lr = 0.00001

    optim = torch.optim.Adamax
    ot = '_Adamax_'

    myDS = MyDataset.CustomImageDataset(cleanedFilePath, imageFolderPath)
    train_test_set = random_split(myDS, [train_ratio, (1 - train_ratio)])

    for model in models:
        queue.put(0)

        pref = sf + model.name + ot + str(lr)
        train_test_set[0].dataset.transform = model.transform
        train_test_set[1].dataset.transform = model.transform

        loader = Loader.create_loaders(train_test_set)

        Trainer.Trainer(model.model, "TorchPretrained", loader, device=device,
                        logger=None, log=False, validation=True,
                        optimizer=optim(model.model.parameters(), lr=lr),
                        loss=nn.CrossEntropyLoss(), epochs=train_length,
                        logSave=pref + '.log',
                        fileSave=pref + '.pth',
                        jobs=queue).start()

    queue.join()
