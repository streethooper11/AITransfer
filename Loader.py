# Based on create_loaders from https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH


from torch.utils.data import DataLoader


def create_loaders(train_set=None, val_set=None, test_set=None, batch_size=64, workers=4, testing=False):
    if testing is True:
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                                drop_last=False)

        loaders = {
            'test': testloader
        }
    else:
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                                 drop_last=True)
        valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                               drop_last=False)

        loaders = {
            'train': trainloader,
            'test': valloader
        }

    return loaders
