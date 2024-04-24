"""https://youtu.be/c36lUUr864M?feature=shared&t=8653"""
import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f"Softmax output np: {outputs}")


x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(f"Softmax output pytorch: {outputs}")

def cross_entropy(y, y_hat):
    return  (- 1 / y.shape[0]) * np.sum(y * np.log(y_hat))

y = np.array([1,0,0])
y_hat = np.array([0.7, 0.2, 0.1])
loss = cross_entropy(y, y_hat)
print(f"Cross-enotropy np: {loss:.3}")

loss = nn.CrossEntropyLoss()  # this already includes nn.LogSoftmax + nn.NLLLoss
y = torch.tensor([0])  # class zero (first label)
y_hat = torch.tensor([[2.0, 1.0, 0.1]])  # size: num_samples * num_classes. Raw values (before softmax)
print(f"Cross-entropy torch: {loss(y_hat, y):.3}")

_, prediction = torch.max(y_hat, dim=1)
print(f"Class with the highest prob: {prediction}")

# with multiple samples
y = torch.tensor([2, 0, 1])
y_hat = torch.tensor([[1.0, 0.1, 3.5], [2.0, 0.1, 0.5], [2.0, 5.0, 0.5]])
print(f"Cross-entropy torch: {loss(y_hat, y):.3}")


class MultiCat(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiCat, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax since we use nn.CrossEntropyLoss()
        return out

model = MultiCat(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()