"""https://youtu.be/c36lUUr864M?si=dxeGkIWfLwULrOso"""

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available else "cpu")
input_size = 784  # 28*28 pixels
hidden_size = 100
num_classes = 10
num_epochs = 4
batch_size = 32
learning_date = 0.001

# MNIST
train = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

# Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples._next_data()
print(f"samples size: {samples.shape}\n")
print(f"labels size: {labels.shape}")

for i in range(6):
    plt.subplot(2,3, i + 1)
    plt.imshow(samples[i][0], cmap="grey")
# plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.l2(out)
        # no softmax here because we use cros-enptropy loss
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_date)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # reshape images to 28*28=input_size
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # forward
        output = model(images)
        loss = criterion(output, labels)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print(f"epoch: {epoch + 1} / {num_epochs}, step: {i + 1} / {n_total_steps}, loss: {loss.item():.4}")

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for (images, labels) in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)  # this is in logit because to softmax is done in optimizer 
        # torch.softmax(outputs,dim=1).sum(dim=1)  # if prob needed
        
        # value, index
        _, prediction = torch.max(outputs, axis=1)  # return with max prob
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()


acc = 100.0 * n_correct / n_samples
print(f"Accuracy={acc}")
