"""https://youtu.be/c36lUUr864M?feature=shared&t=3318"""

import numpy as np

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x
# loss
def loss(y, y_predicted):
    return np.mean((y - y_predicted) ** 2)

# gradient
# MSE = 1/N * (w * x - y)**2
# dJ/dw = 1/N 2x (w * x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print(f"Prediction before training: f(5) {forward(5):.3f}")

# Training
lr = 0.01
n_iters = 20

for epoch in range(n_iters):
    # forward
    y_pred = forward(X)
    # loss
    l = loss(y, y_pred)
    # gradient
    dw = gradient(X, y, y_pred)
    # update weights
    w -= lr * dw

    if epoch % 2 == 0:
        print(f"epoch {epoch + 1}: w={w:.3f}, loss={l:.8f}") 

print(f"Prediction after training: f(5) {forward(5):.3f}")


# WITH TORCH
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

# model prediction
def forward(x):
    return w * x
# loss
def loss(y, y_predicted):
    return ((y - y_predicted) ** 2).mean()

# gradient
# MSE = 1/N * (w * x - y)**2
# dJ/dw = 1/N 2x (w * x - y)

print(f"Prediction before training: f(5) {forward(5):.3f}")

# Training
lr = 0.01
n_iters = 100

for epoch in range(n_iters):
    # forward
    y_pred = forward(X)
    # loss
    l = loss(y, y_pred)
    # gradient
    l.backward()
    # update weights
    with torch.no_grad():  # should not be part of gradient computation
        w -= lr * w.grad
    
    w.grad.zero_()
    
    if epoch % 10 == 0:
        print(f"epoch {epoch + 1}: w={w:.3f}, loss={l:.8f}") 

print(f"Prediction after training: f(5) {forward(5):.3f}")