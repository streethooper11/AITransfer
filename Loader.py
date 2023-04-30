# Based on create_loaders from https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH


from torch.utils.data import DataLoader, random_split


def create_loaders(dataset, train_ratio, batch_size=32, workers=2):
    train_test_set = random_split(dataset, [train_ratio, (1 - train_ratio)])

    trainloader = DataLoader(train_test_set[0], batch_size=batch_size, shuffle=True, num_workers=workers,
                             drop_last=True)
    testloader = DataLoader(train_test_set[1], batch_size=batch_size, shuffle=False, num_workers=workers,
                            drop_last=False)

    loaders = {
        'train': trainloader,
        'test': testloader
    }

    return loaders
