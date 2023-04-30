# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

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

    pretrained_torch_model = torchvision.models.resnet18(weights='DEFAULT')
    # For the ones that use classifier layers
    # print(pretrained_torch_model.classifier[-1])
    # For the ones that use fc as the last layer
    print(pretrained_torch_model.fc)

    ImageNet_transforms = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
    print(ImageNet_transforms)

    # For the ones that use classifier layers
    # pretrained_torch_model.classifier[-1] = nn.Linear(1280, 2)
    # For the ones that use fc as the last layer
    pretrained_torch_model.fc = nn.Linear(in_features=512, out_features=2)

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

    model_save_dir = 'content\\torch_logs'
    tb_writer = SummaryWriter(model_save_dir)
    #    tb_writer.add_graph(pretrained_torch_model.to("cpu"),
    #                        input_to_model=next(iter(loaders['train']))[0][0:])
    tb_writer.close()

    train_length = 10

    # Fine-tuning the ConvNet
    trainer = Trainer.Trainer(pretrained_torch_model, "TorchPretrained", loaders,
                              0.001, device, tb_writer, False)
    trainer.train(epochs=train_length)
