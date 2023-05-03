# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import csv

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

    # model = torchvision.models.vit_b_16(weights='DEFAULT')
    # For the ones that use classifier layers
    # print(model.classifier[-1])
    # For the ones that use fc as the last layer
    # print(model.fc)
    # ImageNet_transforms = torchvision.models.VGG11_Weights.DEFAULT.transforms()
    # print(ImageNet_transforms)

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

    models = []
    loaders = []

    models.append(Models.AlexNet(outputNum))
    models.append(Models.EfficientNetB0(outputNum))
    models.append(Models.InceptionV3(outputNum))
    # models.append(Models.MaxVitT(outputNum))
    # models.append(Models.MNasNet05(outputNum))
    # models.append(Models.MobileNetV3L(outputNum))
    # models.append(Models.RegNetY400MF(outputNum))
    # models.append(Models.ResNet18(outputNum))
    # models.append(Models.ResNet50(outputNum))
    # models.append(Models.ResNext50(outputNum))
    # models.append(Models.ShuffleNetV205(outputNum))
    # models.append(Models.SqueezeNet10(outputNum))
    # models.append(Models.SwinT(outputNum))
    # models.append(Models.VitB16(outputNum))
    # models.append(Models.Vgg(outputNum))

    for model in models:
        loaders.append(
            Loader.create_loaders(
                MyDataset.CustomImageDataset(cleanedFilePath, imageFolderPath, model.transform),
                train_ratio=0.8
            )
        )

    train_length = 15

    sf = 'save\\'
    lr = 0.00001

    optim = torch.optim.Adamax
    ot = '_Adamax_'

    trainers = []

    for i in range(len(models)):
        pref = sf + models[i].name + ot + str(lr)
        trainers.append(
                Trainer.Trainer(models[i].model, "TorchPretrained", loaders[i], device=device,
                                logger=None, log=False, validation=True,
                                optimizer=optim(models[i].model.parameters(), lr=lr),
                                loss=nn.CrossEntropyLoss(), epochs=train_length,
                                logSave=pref + '.log',
                                fileSave=pref + '.pth')
            )

    for t in trainers:
        t.start()

    for t in trainers:
        t.join()
