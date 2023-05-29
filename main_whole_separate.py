# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os
import random

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import Loader
import Models
import MyDataset
import Trainer

import TransformUtil
import DataUtil
import TestUtil

from DiseaseEnum import Disease


def training_stage(device, imagePath, csvpath, savefolder, model_t, optim_t, optim_f,
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

    trainer = Trainer.TrainerSingle(
        None,
        model,
        loader,
        device=device,
        validation=True,
        loss=torch.nn.BCELoss(),
        pref=savefolder,
        name=name,
        ratio=train_ratio
    )

    _ = trainer.train()

    # fine-tune the whole model
    model.define_grads_and_last_layer(feature_extract=False)
    model.updateparams()
    model.model.to(device)
    model.defineOpt(optim_f)
    trainer.pref = os.path.join(savefolder, 'finetune')
    trainer.suffix = model.name + model.opt[1] + model.opt[2] + '_decay_' + \
                     str(model.opt[3]) + '_trainratio_' + str(train_ratio)

    bestmodel = trainer.train()

    return bestmodel


def doOneIter(useset, model_t, optim_t, optim_f, train_t, valid_t,
              resizeflag, usedcolumn, takenum, position):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # training stage; save the model with the highest f1 score
    if resizeflag is True:
        topsetfolder = os.path.join('set', useset, 'resized', '')
        topsavefolder = os.path.join('save', 'resized', useset, column.name, position, '')
    else:
        topsetfolder = os.path.join('set', useset, '')
        topsavefolder = os.path.join('save', useset, column.name, position, '')

    imageFolderPath = os.path.join(topsetfolder, 'train', '')
    sf = os.path.join(topsavefolder, takenum, 'train', '')
    csvfolder = os.path.join(topsavefolder, 'train', '')

    os.makedirs(sf, exist_ok=True)

    # making clean csv files
    # DataUtil.makecleanedcsv(csvfolder, imageFolderPath, setname, usedcolumn.value, position)
    # csvpath = os.path.join(csvfolder, 'Entry_cleaned.csv')
    # DataUtil.savestratifiedsplits(csvpath, csvfolder, 0.8)

    bestmodel = training_stage(device, imageFolderPath, csvfolder, sf, model_t, optim_t,
                               optim_f, outputNum=1, train_ratio=0.8, name=usedcolumn.name,
                               train_t=train_t, valid_t=valid_t)

    # test each iteration
    TestUtil.test_stage(bestmodel, device, topsetfolder, topsavefolder, takenum, model_t, optim_f, valid_t,
                        usedcolumn)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    setname = 'whole'

    modeltype = Models.MobileNetV2
    optim_transfer = (torch.optim.Adam, '_Adam_', '0.001', 0.0, 30)
    optim_finetuning = (torch.optim.SGD, '_SGD_fine_', '0.00001', 0.0, 25)

    resized = True  # True means use previously resized images
    column = Disease.HasDisease  # Used when multi-label flag is false; only work on this column
    view_position = ''

    if resized is True:
        topset = os.path.join('set', 'resized', setname, '')
        topsave = os.path.join('save', 'resized', setname, column.name, view_position, '')
    else:
        topset = os.path.join('set', setname, '')
        topsave = os.path.join('save', setname, column.name, view_position, '')

    augment_options = [
        '000', '001', '002', '010', '011', '012', '100', '101', '102', '110', '111', '112'
    ]

    transform_options = [
        '000', '010', '020', '100', '110', '120'
    ]

    random.shuffle(augment_options)

    for a in augment_options:
        random.shuffle(transform_options)
        for t in transform_options:
            foldername = a + '_' + t
            train_transform, valid_transform = TransformUtil.getTransforms(a, t, resized)
            doOneIter(setname, modeltype, optim_transfer, optim_finetuning,
                      train_transform, valid_transform, resized, column, foldername, view_position)

            transformsave = os.path.join(topsave, foldername, 'Transformations.txt')
            with open(transformsave, 'w') as logFile:
                logFile.write('Train transform:\n')
                logFile.write(str(train_transform))
                logFile.write('\nValidation transform:\n')
                logFile.write(str(valid_transform))
