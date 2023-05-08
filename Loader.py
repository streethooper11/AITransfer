# Based on create_loaders from https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH


from torch.utils.data import DataLoader


def create_loaders(train_set, test_set, batch_size=32, workers=2):
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                             drop_last=True)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                            drop_last=False)

    loaders = {
        'train': trainloader,
        'test': testloader
    }

    return loaders
