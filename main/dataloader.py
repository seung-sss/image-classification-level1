import torch
from torch.utils.data import DataLoader

def getDataloader(dataset, batch_size, shuffle, num_workers):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return data_loader