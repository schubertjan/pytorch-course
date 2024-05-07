"""https://youtu.be/c36lUUr864M?si=PqB5qyjYpIdT0_n0&t=11657"""

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available else "cpu")

num_epochs = 4
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # n channels = output size of conv1
        self.fc1 = nn.Linear(16*5*5, 100)  # flatten
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 different classes
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # no softmax here because we use cros-enptropy loss
        return x
    
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        output = model(images)
        loss = criterion(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"epoch: {epoch + 1} / {num_epochs}, step: {i + 1} / {len(train_loader)}, loss: {loss.item():.4}")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for (images, labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # this is in logit because to softmax is done in optimizer 
        # torch.softmax(outputs,dim=1).sum(dim=1)  # if prob needed
        
        # value, index
        _, prediction = torch.max(outputs, axis=1)  # return with max prob
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()


acc = 100.0 * n_correct / n_samples
print(f"Accuracy={acc}")