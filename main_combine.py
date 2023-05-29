# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os
import sys
import random

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


def makecleanedcsv(topsavefolder, imageFolderPath, usedset, colenum, position):
    if usedset == 'sample':
        origcsvpath = os.path.join('set', 'sample_labels.csv')
    else:
        origcsvpath = os.path.join('set', 'Data_Entry_2017.csv')

    cleanpath = os.path.join(topsavefolder, 'Entry_cleaned.csv')
    clean_then_save_csv(origcsvpath, cleanpath, imageFolderPath, colenum, position)


def training_stage(device, imagePath, csvpath, savefolder, bestmodel, model_t, optim_t,
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
    if bestmodel is not None:
        checkpoint = torch.load(bestmodel[0])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.opt[0].load_state_dict(checkpoint['optimizer_state_dict'])

    trainer = Trainer_combine.TrainerCombineSingle(
        bestmodel,
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


def doOneIter(bestmodel, allsets, model_t, optim_t, train_t, valid_t, resizeflag, usedcolumn, takenum, position):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training stage; save the model with the highest f1 score

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
        # makecleanedcsv(csvfolder, imageFolderPath, setname, usedcolumn.value, position)
        # csvpath = os.path.join(csvfolder, 'Entry_cleaned.csv')
        # savestratifiedsplits(csvpath, csvfolder, 0.8)

        bestmodel = training_stage(device, imageFolderPath, csvfolder, sf, bestmodel, model_t, optim_t,
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
    checkpoint = torch.load(bestmodel[0])
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.opt[0].load_state_dict(checkpoint['optimizer_state_dict'])

    trainer = Trainer_combine.TrainerCombineSingle(
        best_path=None,
        modelinf=model,
        loaders=loader,
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

    return bestmodel


def getTransforms(a_opt, t_opt, resizeflag):
    translate_per = dict()
    translate_per['x'] = (-0.1, 0.1)
    translate_per['y'] = (-0.1, 0)

    augment1_need1 = A.Affine(scale=None, translate_percent=None, rotate=(-10, 10), p=1.0)
    augment1_need2 = A.Affine(scale=None, translate_percent=translate_per, rotate=(-10, 10), p=1.0)
    augment2_option = A.GaussNoise(per_channel=False, p=1.0)
    augment3_option1 = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
    augment3_option2 = A.ColorJitter(brightness=0.1, contrast=0.1)

    transform1_option = A.ToSepia(p=1.0)
    transform2_option1 = A.CLAHE(p=1.0)
    transform2_option2 = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)
    transform3_need1 = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    transform3_need2 = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augment4_always = A.HorizontalFlip(p=0.5)
    transform4_always = A.pytorch.transforms.ToTensorV2()

    resizing_transform = A.Resize(224, 224)

    if resizeflag:
        prefix = []
        suffix = [transform4_always]
    else:
        prefix = [A.CenterCrop(width=850, height=850, p=1.0)]
        suffix = [resizing_transform, transform4_always]

    augments = [augment4_always]
    transforms = []

    if a_opt[0] == '0':
        augments.append(augment1_need1)
    else:
        augments.append(augment1_need2)

    if a_opt[1] == '1':
        augments.append(augment2_option)

    if a_opt[2] == '1':
        augments.append(augment3_option1)
    elif a_opt[2] == '2':
        augments.append(augment3_option2)

    if t_opt[0] == '1':
        transforms.append(transform1_option)

    if t_opt[1] == '1':
        transforms.append(transform2_option1)
    elif t_opt[1] == '2':
        transforms.append(transform2_option2)

    if t_opt[2] == '0':
        transforms.append(transform3_need1)
    else:
        transforms.append(transform3_need2)

    train_transform = A.Compose(
        prefix + augments + transforms + suffix
    )
    test_transform = A.Compose(
        prefix + transforms + suffix
    )

    return train_transform, test_transform


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    setnames = [
        # 'set2',
        # 'set3',
        # 'set4',
        # 'set5',
        # 'set6',
        # 'set7',
        # 'set8',
        # 'set9',
        # 'set10',
        # 'set11',
        'sample',
    ]

    modeltype = Models.MobileNetV2
    optim = (torch.optim.Adam, '_Adam_', '0.0001', 0.0, 20)

    column = Disease.HasDisease  # Used when multi-label flag is false; only work on this column
    view_position = ''

    if len(sys.argv) > 3:
        if sys.argv[1] == '0':
            resized = False
        else:
            resized = True

        augment_options = sys.argv[2]
        transform_options = sys.argv[3]
    else:
        resized = True
        augment_options = '000'
        transform_options = '120'

    foldername = augment_options + '_' + transform_options
    train_transform, valid_transform = getTransforms(augment_options, transform_options, resized)

    augment_options = [
        '000', '001', '002', '010', '011', '012', '100', '101', '102', '110', '111', '112'
    ]

    transform_options = [
        '000', '010', '020', '100', '110', '120'
    ]

    random.shuffle(augment_options)

    bestmodelinf = None
    for a in augment_options:
        random.shuffle(transform_options)
        for t in transform_options:
            foldername = a + '_' + t
            train_transform, valid_transform = getTransforms(a, t, resized)
            bestmodelinf = doOneIter(bestmodelinf, setnames, modeltype, optim,
                                      train_transform, valid_transform, resized, column, foldername, view_position)

            if resized is True:
                topsave = os.path.join('save', 'resized', 'combine', column.name, view_position, '')
            else:
                topsave = os.path.join('save', 'combine', column.name, view_position, '')

            transformsave = os.path.join(topsave, foldername, 'Transformations.txt')
            with open(transformsave, 'w') as logFile:
                logFile.write('Train transform:\n')
                logFile.write(str(train_transform))
                logFile.write('\nValidation transform:\n')
                logFile.write(str(valid_transform))

"""
    bestmodelpath = doOneIter(bestmodelpath, setnames, modeltype, optim, train_transform, valid_transform, resized,
                              column, foldername, view_position)

    if resized is True:
        topsave = os.path.join('save', 'resized', 'combine', column.name, view_position, '')
    else:
        topsave = os.path.join('save', 'combine', column.name, view_position, '')

    transformsave = os.path.join(topsave, foldername, 'Transformations.txt')
    with open(transformsave, 'w') as logFile:
        logFile.write('Train transform:\n')
        logFile.write(str(train_transform))
        logFile.write('\nValidation transform:\n')
        logFile.write(str(valid_transform))
"""
