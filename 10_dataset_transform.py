"""https://youtu.be/c36lUUr864M?feature=shared&t=8007"""
# https://raw.githubusercontent.com/patrickloeber/pytorchTutorial/master/data/wine/wine.csv
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class Wine(Dataset):
    def __init__(self, transform=None) -> None:
        xy = np.loadtxt("https://raw.githubusercontent.com/patrickloeber/pytorchTutorial/master/data/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_samples

class ToTensor():
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        return inputs * self.factor, target

wine = Wine(transform=ToTensor())
first_data = wine[0]
features, labels = first_data

print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
wine = Wine(composed)
print(wine[0])