# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import csv
import albumentations as A
import albumentations.pytorch

import Loader
import Models
import MyDataset
import Trainer_combine
from DiseaseEnum import Disease


def savestratifiedsplits(dataPath, savefolder, train_ratio):
    df = pd.read_csv(dataPath)

    y = df.iloc[:, -1]
    train, test = train_test_split(df, train_size=train_ratio, stratify=y, random_state=11)

    train.to_csv(os.path.join(savefolder, 'Entry_cleaned_Train.csv'), index=False)
    test.to_csv(os.path.join(savefolder, 'Entry_cleaned_Test.csv'), index=False)


def clean_then_save_csv(origFilePath, cleanFilePath, imgPath, colenum, position):
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
                        if position != '':
                            if row[6] == position:
                                if os.path.exists(img_path):
                                    if row[1] == "No Finding":
                                        csv_write.writerow([row[0], 0])
                                    else:
                                        csv_write.writerow([row[0], 1])
                        else:
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


def makecleanedcsv(topsavefolder, imageFolderPath, colenum, position):
    origcsvpath = os.path.join('set', 'Data_Entry_2017.csv')
    cleanpath = os.path.join(topsavefolder, 'Entry_cleaned.csv')
    clean_then_save_csv(origcsvpath, cleanpath, imageFolderPath, colenum, position)


def training_stage(device, imagePath, csvpath, savefolder, best_path, model_t, optim_t,
                   outputNum=1, train_ratio=0.8, name='', train_t=None, valid_t=None):
    train_path = os.path.join(csvpath, 'Entry_cleaned_Train.csv')
    train_df = pd.read_csv(train_path)
    x_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]

    trainDS = MyDataset.CustomImageDatasetSingle(imagePath, x_train, y_train)
    trainDS.transform = train_t

    valid_path = os.path.join(csvpath, 'Entry_cleaned_Test.csv')
    valid_df = pd.read_csv(valid_path)
    x_val = valid_df.iloc[:, 0:-1]
    y_val = valid_df.iloc[:, -1]

    valDS = MyDataset.CustomImageDatasetSingle(imagePath, x_val, y_val)
    valDS.transform = valid_t

    loader = Loader.create_loaders(train_set=trainDS, val_set=valDS)

    model = model_t(outputNum, device, optim_t)
    if best_path is not None:
        checkpoint = torch.load(best_path)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.opt[0].load_state_dict(checkpoint['optimizer_state_dict'])

    trainer = Trainer_combine.TrainerCombineSingle(
        model,
        loader,
        device=device,
        validation=True,
        loss=torch.nn.BCELoss(),
        pref=savefolder,
        name=name,
        ratio=train_ratio
    )

    best_model = trainer.train()

    return best_model


def doOneIter(device, allsets, model_t, optim_t, train_t, valid_t, resizeflag, usedcolumn, takenum, position):
    # training stage; save the model with the highest f1 score
    bestmodelpath = None

    if resizeflag is True:
        topsetfolder = os.path.join('set', 'all', 'resized', '')
        topsavefolder = os.path.join('save', 'resized', 'combine', column.name, position, '')
    else:
        topsetfolder = os.path.join('set', 'all', '')
        topsavefolder = os.path.join('save', 'combine', column.name, position, '')

    for setname in allsets:
        imageFolderPath = os.path.join(topsetfolder, setname, '')
        sf = os.path.join(topsavefolder, takenum, setname, '')
        csvfolder = os.path.join(topsavefolder, setname, '')

        os.makedirs(sf, exist_ok=True)

        # making clean csv files
        # makecleanedcsv(sf, imageFolderPath, usedcolumn.value, position)
        # csvpath = os.path.join(csvfolder, 'Entry_cleaned.csv')
        # savestratifiedsplits(csvpath, csvfolder, 0.8)

        bestmodelpath = training_stage(device, imageFolderPath, csvfolder, sf, bestmodelpath, model_t, optim_t,
                                       outputNum=1, train_ratio=0.8, name=usedcolumn.name,
                                       train_t=train_t, valid_t=valid_t)

    # test stage
    imageFolderPath = os.path.join(topsetfolder, 'test', '')
    sf = os.path.join(topsavefolder, takenum, 'test', '')
    csvfolder = os.path.join(topsavefolder, 'test', '')

    os.makedirs(sf, exist_ok=True)
    testcsvpath = os.path.join(csvfolder, 'Entry_cleaned.csv')

    test_df = pd.read_csv(testcsvpath)
    x = test_df.iloc[:, 0:-1]
    y = test_df.iloc[:, -1]

    ds = MyDataset.CustomImageDatasetSingle(imageFolderPath, x, y)
    ds.transform = valid_t

    loader = Loader.create_loaders(test_set=ds, testing=True)

    model = model_t(1, device, optim_t)
    checkpoint = torch.load(bestmodelpath)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.opt[0].load_state_dict(checkpoint['optimizer_state_dict'])

    trainer = Trainer_combine.TrainerCombineSingle(
        model,
        loader,
        device=device,
        validation=True,
        loss=torch.nn.BCELoss(),
        pref=sf,
        name=usedcolumn.name,
        ratio=0.8
    )

    logsave = os.path.join(trainer.pref, 'logs', trainer.suffix + '.log')
    os.makedirs(os.path.dirname(logsave), exist_ok=True)
    with open(logsave, 'w') as logFile:
        trainer.evaluate(logFile=logFile)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    usedevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setnames = [
        'set2',
        'set3',
        'set4',
        'set5',
        'set6',
        'set7',
        'set8',
        'set9',
        'set10',
        'set11',
    ]

    translate_per = dict()
    translate_per['x'] = (-0.1, 0.1)
    translate_per['y'] = (-0.1, 0)

    modeltype = Models.MobileNetV2
    optim = (torch.optim.Adam, '_Adam_', '0.0001', 0.0, 30)

    order1_augment_always = A.HorizontalFlip(p=0.5)
    order2_augment_option1 = A.Affine(scale=None, translate_percent=translate_per, rotate=(-10, 10), p=1.0)
    order2_augment_option2 = A.Affine(scale=None, translate_percent=None, rotate=(-10, 10), p=1.0)
    order3_augment_option = A.GaussNoise(per_channel=False, p=0.5)
    order4_augment_option1 = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
    order4_augment_option2 = A.ColorJitter(brightness=0.1, contrast=0.1)
    order5_transform_resize = A.Resize(224, 224)
    order6_transform_option1 = A.CLAHE(p=1.0)
    order6_transform_option2 = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)
    order7_transform_option1 = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    order7_transform_option2 = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    order8_transform_always = A.pytorch.transforms.ToTensorV2()

    train_transforms = {
        0: [order2_augment_option1],
        1: [order2_augment_option1, order6_transform_option1],
        2: [order2_augment_option1, order6_transform_option2],
        3: [order2_augment_option1, order4_augment_option1],
        4: [order2_augment_option1, order4_augment_option1, order6_transform_option1],
        5: [order2_augment_option1, order4_augment_option1, order6_transform_option2],
        6: [order2_augment_option1, order4_augment_option2],
        7: [order2_augment_option1, order4_augment_option2, order6_transform_option1],
        8: [order2_augment_option1, order4_augment_option2, order6_transform_option2],
        9: [order2_augment_option1, order3_augment_option],
        10: [order2_augment_option1, order3_augment_option, order6_transform_option1],
        11: [order2_augment_option1, order3_augment_option, order6_transform_option2],
        12: [order2_augment_option1, order3_augment_option, order4_augment_option1],
        13: [order2_augment_option1, order3_augment_option, order4_augment_option1, order6_transform_option1],
        14: [order2_augment_option1, order3_augment_option, order4_augment_option1, order6_transform_option2],
        15: [order2_augment_option1, order3_augment_option, order4_augment_option2],
        16: [order2_augment_option1, order3_augment_option, order4_augment_option2, order6_transform_option1],
        17: [order2_augment_option1, order3_augment_option, order4_augment_option2, order6_transform_option2],
        18: [order2_augment_option2],
        19: [order2_augment_option2, order6_transform_option1],
        20: [order2_augment_option2, order6_transform_option2],
        21: [order2_augment_option2, order4_augment_option1],
        22: [order2_augment_option2, order4_augment_option1, order6_transform_option1],
        23: [order2_augment_option2, order4_augment_option1, order6_transform_option2],
        24: [order2_augment_option2, order4_augment_option2],
        25: [order2_augment_option2, order4_augment_option2, order6_transform_option1],
        26: [order2_augment_option2, order4_augment_option2, order6_transform_option2],
        27: [order2_augment_option2, order3_augment_option],
        28: [order2_augment_option2, order3_augment_option, order6_transform_option1],
        29: [order2_augment_option2, order3_augment_option, order6_transform_option2],
        30: [order2_augment_option2, order3_augment_option, order4_augment_option1],
        31: [order2_augment_option2, order3_augment_option, order4_augment_option1, order6_transform_option1],
        32: [order2_augment_option2, order3_augment_option, order4_augment_option1, order6_transform_option2],
        33: [order2_augment_option2, order3_augment_option, order4_augment_option2],
        34: [order2_augment_option2, order3_augment_option, order4_augment_option2, order6_transform_option1],
        35: [order2_augment_option2, order3_augment_option, order4_augment_option2, order6_transform_option2],
    }

    valid_transforms = {
        0: [],
        1: [order6_transform_option1],
        2: [order6_transform_option2],
    }

    train_prefix = [order1_augment_always]
    transform_suffix = [order7_transform_option2, order8_transform_always]
    if len(sys.argv) > 1:
        option = int(sys.argv[1])
    else:
        option = 12

    resized = True  # True means use previously resized images
    column = Disease.HasDisease  # Used when multi-label flag is false; only work on this column
    view_position = ''

    while option < 15:
        train_transform = A.Compose(
            train_prefix + train_transforms[option] + transform_suffix
        )

        valid_transform = A.Compose(
            valid_transforms[option % 3] + transform_suffix
        )

        doOneIter(usedevice, setnames, modeltype, optim, train_transform, valid_transform, resized,
                  column, str(option), view_position)

        if resized is True:
            topsave = os.path.join('save', 'resized', 'combine', column.name, view_position, '')
        else:
            topsave = os.path.join('save', 'combine', column.name, view_position, '')

        transformsave = os.path.join(topsave, str(option), 'Transformations.txt')
        with open(transformsave, 'w') as logFile:
            logFile.write('Train transform:\n')
            logFile.write(str(train_transform))
            logFile.write('\nValidation transform:\n')
            logFile.write(str(valid_transform))

        option += 1
