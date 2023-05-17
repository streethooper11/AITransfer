# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH
import os

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import csv

import Loader
import Models
import MyDataset
import Tester
from DiseaseEnum import Disease


def clean_then_save_csv(origFilePath, cleanFilePath, imgPath):
    with open(origFilePath, newline='') as csvreadfile:
        with open(cleanFilePath, 'w', newline='') as csvwritefile:
            csv_read = csv.reader(csvreadfile, delimiter=',')
            csv_write = csv.writer(csvwritefile, delimiter=',')
            firstrow = True

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


if __name__ == "__main__":
    # variables you can change
    setname = 'set3'  # choose the set to work on
    modelfolder = 'save\\' + setname + '\\multiclass\\models\\'
#    modelpath = modelfolder + 'f1_0.2946MobileNetV2_Adam_0.0001_decay_0.0_trainratio_0.8_epoch19.pth'
    modelpath = modelfolder + 'f1_0.2932MobileNetV2_Adam_0.0003_decay_0.0_trainratio_0.8_epoch36.pth'

    learning_rate = '0.0003'

    originalFilePath = 'set\\Data_Entry_2017.csv'

    imageFolderPath = 'set\\' + 'set3' + '\\test\\'

    sf = 'save\\' + setname + '\\multiclass\\test\\'

    outputNum = len(Disease) - 1

    os.makedirs(sf, exist_ok=True)
    cleanedFilePath = sf + 'Entry_cleaned.csv'
    clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Models.MobileNetV2(14, device, (torch.optim.Adam, '_Adam_', learning_rate, 0.0, 10))
    model.model.load_state_dict(torch.load(modelpath))

    df = pd.read_csv(cleanedFilePath)
    x = df.iloc[:, 0:-outputNum]
    y = torch.tensor(df.iloc[:, -outputNum:].values)
    y = y.type(torch.float)

    testDS = MyDataset.CustomImageDatasetMulti(imageFolderPath, x, y)
    testDS.transform = model.transform
    loader = Loader.create_loaders(test_set=testDS, testing=True)

    suffix = model.name + model.opt[1] + model.opt[2] + '_test'

    tester = Tester.Tester(
        model.model,
        loader,
        device=device,
        loss=torch.nn.BCELoss(),
        pref=sf,
        suffix=suffix,
        numoutputs=outputNum
    )

    tester.test()
