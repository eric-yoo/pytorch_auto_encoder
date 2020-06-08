import glob
import os

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import random_split

def get_loader(dataroot, batch_size, shuffle, transform, num_workers):
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)
        print('[*] Make dataroot directory!')

    train_set = MNIST(dataroot, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    test_set = MNIST(dataroot, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
