# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import copy
import os
import sys
import random

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import albumentations.pytorch

import Loader
import Models
import MyDataset
import Trainer

import DataUtil
import TestUtil
import TransformUtil

from DiseaseEnum import Disease


def training_stage(device, imagePath, csvpath, savefolder, bestmodel, model_t, optim_t, optim_f,
                   outputNum=1, train_ratio=0.8, name='', train_t=None, valid_t=None):
    train_path = os.path.join(csvpath, 'Entry_cleaned_2020_Train.csv')
    train_df = pd.read_csv(train_path)
    x_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]

    trainDS = MyDataset.CustomImageDatasetSingle(imagePath, x_train, y_train)
    trainDS.transform = train_t

    valid_path = os.path.join(csvpath, 'Entry_cleaned_2020_Test.csv')
    valid_df = pd.read_csv(valid_path)
    x_val = valid_df.iloc[:, 0:-1]
    y_val = valid_df.iloc[:, -1]

    valDS = MyDataset.CustomImageDatasetSingle(imagePath, x_val, y_val)
    valDS.transform = valid_t

    loader = Loader.create_loaders(train_set=trainDS, val_set=valDS)

    # make the model with frozen layers and a new classification layer; load a previous model if given
    model = model_t(outputNum, device, optim_t, feature_extract=True)
    if bestmodel is not None:
        checkpoint = torch.load(bestmodel[0])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.opt[0].load_state_dict(checkpoint['optimizer_state_dict'])

    # train the new layer
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

    # define a new model; fine-tune from the best checkpoint
    model_fine = model_t(outputNum, device, optim_f, feature_extract=False)

    checkpoint = torch.load(bestmodel[0])
    model_fine.model.load_state_dict(checkpoint['model_state_dict'])

    trainer_fine = Trainer.TrainerSingle(
        bestmodel,
        model_fine,
        loader,
        device=device,
        validation=True,
        loss=torch.nn.BCELoss(),
        pref=os.path.join(savefolder, 'finetune'),
        name=name,
        ratio=train_ratio
    )

    bestmodel = trainer_fine.train()

    return bestmodel


def doOneIter(bestmodel, allsets, model_t, optim_t, optim_f,
              train_t, valid_t, resizeflag, usedcolumn, takenum, position, uniqueflag):

    if len(sys.argv) > 3 and (sys.argv[2] == 'cpu' or sys.argv[2] == 'cuda'):
        device = sys.argv[2]
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # training stage; save the model with the highest f1 score
    if resizeflag:
        topsetfolder = os.path.join('set', 'all', 'resized', '')

        if uniqueflag:
            topsavefolder = os.path.join('save', 'resized', 'combine', usedcolumn.name, 'unique', position, '')
        else:
            topsavefolder = os.path.join('save', 'resized', 'combine', usedcolumn.name, position, '')
    else:
        topsetfolder = os.path.join('set', 'all', '')

        if uniqueflag:
            topsavefolder = os.path.join('save', 'combine', usedcolumn.name, 'unique', position, '')
        else:
            topsavefolder = os.path.join('save', 'combine', usedcolumn.name, position, '')

    for setname in allsets:
        imageFolderPath = os.path.join(topsetfolder, setname, '')
        sf = os.path.join(topsavefolder, takenum, setname, '')
        csvfolder = os.path.join(topsavefolder, setname, '')

        os.makedirs(sf, exist_ok=True)

        # making clean csv files
        DataUtil.makecleanedcsv(csvfolder, imageFolderPath, setname, usedcolumn.value, position, uniqueflag)
        csvpath = os.path.join(csvfolder, 'Entry_cleaned_2020.csv')
        DataUtil.savestratifiedsplits(csvpath, csvfolder, 0.8)

        bestmodel = training_stage(device, imageFolderPath, csvfolder, sf, bestmodel, model_t, optim_t,
                                   optim_f, outputNum=1, train_ratio=0.8, name=usedcolumn.name,
                                   train_t=train_t, valid_t=valid_t)
        # Test each iteration
        TestUtil.test_stage(bestmodel, device, topsetfolder, topsavefolder, sf, model_t, optim_f, valid_t,
                            usedcolumn)



def oneBigTempIter(sets, usedcolumn, position, resizeflag, normflag, sepiaflag, sharpenopt, foldername,
                   gaussflag, scaleflag, uniqueflag):
    train_transform, valid_transform = TransformUtil.getTransformsFromFlags(
        resizeflag, normflag, sepiaflag, sharpenopt, gaussflag, scaleflag)

    modeltype = Models.MobileNetV2
    optim_transfer = (torch.optim.Adam, '_Adam_', '0.001', 0.0, 50)
    optim_finetuning = (torch.optim.Adam, '_Adam_fine_', '0.0001', 0.0, 50)

    doOneIter(None, sets, modeltype, optim_transfer, optim_finetuning,
              train_transform, valid_transform, resizeflag, usedcolumn, foldername, position, uniqueflag)

    if resizeflag is True:
        if uniqueflag:
            topsave = os.path.join('save', 'resized', 'combine', usedcolumn.name, 'unique', position, '')
        else:
            topsave = os.path.join('save', 'resized', 'combine', usedcolumn.name, position, '')
    else:
        if uniqueflag:
            topsave = os.path.join('save', 'combine', usedcolumn.name, 'unique', position, '')
        else:
            topsave = os.path.join('save', 'combine', usedcolumn.name, position, '')

    transformsave = os.path.join(topsave, foldername, 'Transformations.txt')
    with open(transformsave, 'w') as logFile:
        logFile.write('Train transform:\n')
        logFile.write(str(train_transform))
        logFile.write('\nValidation transform:\n')
        logFile.write(str(valid_transform))


def setFlagsAndDoIter(sets, usedcolumn, position, resizeflag, uniqueflag, transform_opt):
    if transform_opt % 2 == 0:
        norm_imagenet = False
        norm_name = '_graynorm'
    else:
        norm_imagenet = True
        norm_name = '_imagenorm'

    use_sharpen = (transform_opt % 6) // 2
    if use_sharpen == 0:
        sharpen_name = '_nosharp'
    elif use_sharpen == 1:
        sharpen_name = '_CLAHE'
    else:
        sharpen_name = '_sharpen'

    sepia = (transform_opt % 12) // 6
    if sepia == 0:
        use_sepia = False
        sepia_name = '_nosepia'
    else:
        use_sepia = True
        sepia_name = '_sepia'

    scaling = (transform_opt % 24) // 12
    if scaling == 0:
        use_scale = False
        scale_name = '_noscale'
    else:
        use_scale = True
        scale_name = '_scale'

    gaussing = transform_opt // 24
    if gaussing == 0:
        use_gauss = False
        gauss_name = '_nogauss'
    else:
        use_gauss = True
        gauss_name = '_gauss'

    folder_name_use = 'custom' + gauss_name + scale_name + sepia_name + sharpen_name + norm_name

    oneBigTempIter(sets, usedcolumn, position, resizeflag, norm_imagenet, use_sepia,
                   use_sharpen, folder_name_use, use_gauss, use_scale, uniqueflag)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    sets_to_use = [
        #'set2',
        #'set3',
        #'set4',
        #'set5',
        #'set6',
        #'set7',
        #'set8',
        #'set9',
        #'set10',
        #'set11',
        'whole',
    ]

    column_output = Disease.HasDisease  # Used when multi-label flag is false; only work on this column
    viewing_position = ''

    if len(sys.argv) > 1:
        option = int(sys.argv[1])
    else:
        option = 0

    if len(sys.argv) > 2:
        use_resized = bool(int(sys.argv[2]))
    else:
        use_resized = True

    if len(sys.argv) > 3:
        use_unique = bool(int(sys.argv[3]))
    else:
        use_unique = False

    if option % 2 == 0:
        norm_imagenet = False
        norm_name = '_graynorm'
    else:
        norm_imagenet = True
        norm_name = '_imagenorm'

    use_sharpen = (option % 6) // 2
    if use_sharpen == 0:
        sharpen_name = '_nosharp'
    elif use_sharpen == 1:
        sharpen_name = '_CLAHE'
    else:
        sharpen_name = '_sharpen'

    sepia = (option % 12) // 6
    if sepia == 0:
        use_sepia = False
        sepia_name = '_nosepia'
    else:
        use_sepia = True
        sepia_name = '_sepia'

    scaling = (option % 24) // 12
    if scaling == 0:
        use_scale = False
        scale_name = '_noscale'
    else:
        use_scale = True
        scale_name = '_scale'

    gaussing = option // 24
    if gaussing == 0:
        use_gauss = False
        gauss_name = '_nogauss'
    else:
        use_gauss = True
        gauss_name = '_gauss'

    folder_name_use = 'custom' + gauss_name + scale_name + sepia_name + sharpen_name + norm_name

    oneBigTempIter(sets_to_use, column_output, viewing_position, use_resized, norm_imagenet, use_sepia,
                   use_sharpen, folder_name_use, use_gauss, use_scale, use_unique)

    use_unique = True

    oneBigTempIter(sets_to_use, column_output, viewing_position, use_resized, norm_imagenet, use_sepia,
                   use_sharpen, folder_name_use, use_gauss, use_scale, use_unique)
