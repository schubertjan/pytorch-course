"""https://youtu.be/c36lUUr864M?feature=shared&t=7075"""
# https://raw.githubusercontent.com/patrickloeber/pytorchTutorial/master/data/wine/wine.csv
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class Wine(Dataset):
    def __init__(self) -> None:
        xy = np.loadtxt("https://raw.githubusercontent.com/patrickloeber/pytorchTutorial/master/data/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
wine = Wine()
wine[0]  # print first row

# look at one batch
batch_size = 4
dataloader = DataLoader(dataset=wine, batch_size=batch_size, shuffle=True, num_workers=4)
dataiter = iter(dataloader)
data = dataiter._next_data()
print(data)

# Dummy training loop
n_epochs = 2
total_samples = len(wine)
n_iterations = np.ceil(total_samples / batch_size)

for epoch in range(n_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward
        # backward
        # update weights
        if (i + 1) % 5 == 0:
            print(f"Epoch {(epoch + 1) / n_epochs}, step={(i + 1)}/{n_iterations}, inputs= {inputs.shape}")