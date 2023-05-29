# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os
import sys
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
from DiseaseEnum import Disease


def training_stage(device, imagePath, csvpath, savefolder, bestmodel, model_t, optim_t, optim_f,
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

    trainer = Trainer.TrainerSingle(
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

    bestmodel = trainer.train()

    model.define_grads_and_last_layer(feature_extract=False)
    model.model.to(device)
    model.defineOpt(optim_f)

    trainer = Trainer.TrainerSingle(
        bestmodel,
        model,
        loader,
        device=device,
        validation=True,
        loss=torch.nn.BCELoss(),
        pref=os.path.join(savefolder, 'finetune'),
        name=name,
        ratio=train_ratio
    )

    best_model = trainer.train()

    return best_model


def doOneIter(bestmodel, allsets, model_t, optim_t, optim_f,
              train_t, valid_t, resizeflag, usedcolumn, takenum, position):
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
        # DataUtil.makecleanedcsv(csvfolder, imageFolderPath, setname, usedcolumn.value, position)
        # csvpath = os.path.join(csvfolder, 'Entry_cleaned.csv')
        # DataUtil.savestratifiedsplits(csvpath, csvfolder, 0.8)

        bestmodel = training_stage(device, imageFolderPath, csvfolder, sf, bestmodel, model_t, optim_t,
                                   optim_f, outputNum=1, train_ratio=0.8, name=usedcolumn.name,
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

    trainer = Trainer.TrainerSingle(
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
    optim_transfer = (torch.optim.SGD, '_SGD_', '0.001', 0.0, 10)
    optim_finetuning = (torch.optim.SGD, '_SGD_fine_', '0.00001', 0.0, 10)

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
        transform_options = '000'

    foldername = augment_options + '_' + transform_options
    train_transform, valid_transform = TransformUtil.getTransforms(augment_options, transform_options, resized)

    bestmodelinf = None
    bestmodelinf = doOneIter(bestmodelinf, setnames, modeltype, optim_transfer, optim_finetuning,
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
