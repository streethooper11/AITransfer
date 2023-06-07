import os

import torch
import pandas as pd

import MyDataset
import Loader
import Trainer

def test_stage(bestmodel, device, topsetfolder, topsavefolder, foldname, modeltype, optimtype, test_t,
               column):
    # test stage
    imageFolderPath = os.path.join(topsetfolder, 'test', '')
    sf = os.path.join(topsavefolder, foldname, 'test', '')
    csvfolder = os.path.join(topsavefolder, 'test', '')

    os.makedirs(sf, exist_ok=True)
    testcsvpath = os.path.join(csvfolder, 'Entry_cleaned_2020.csv')

    test_df = pd.read_csv(testcsvpath)
    x = test_df.iloc[:, 0:-1]
    y = test_df.iloc[:, -1]

    ds = MyDataset.CustomImageDatasetSingle(imageFolderPath, x, y)
    ds.transform = test_t

    loader = Loader.create_loaders(test_set=ds, testing=True)

    model = modeltype(1, device, optimtype)
    checkpoint = torch.load(bestmodel[0])
    model.model.load_state_dict(checkpoint['model_state_dict'])

    trainer = Trainer.TrainerSingle(
        best_path=None,
        modelinf=model,
        loaders=loader,
        device=device,
        validation=True,
        loss=torch.nn.BCELoss(),
        pref=sf,
        name=column.name,
        ratio=0.8
    )

    logsave = os.path.join(trainer.pref, 'logs', trainer.suffix + '.log')
    os.makedirs(os.path.dirname(logsave), exist_ok=True)
    with open(logsave, 'w') as logF:
        trainer.evaluate(logFile=logF)
