"""https://youtu.be/c36lUUr864M?feature=shared&t=5234"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], -1)

n_samples, n_features = X.shape

# Model
input_size = n_features
output_size = y.shape[1]
model = nn.Linear(input_size, output_size)

# Loss and optimizer
lr = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # forward
    y_pred = model(X)

    # backward
    l = loss(y_pred, y)
    l.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, loss={l.item():.4}")

predicted = model(X).detach()  # prevent from being tracked in the tensor graph so this will generate a new tensor
predicted = predicted.numpy()
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()