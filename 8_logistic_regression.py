"""https://youtu.be/c36lUUr864M?feature=shared&t=5968"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

# Model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression(n_features)
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_epochs = 1000
for epoch in range(n_epochs):
    y_pred = model(X_train)
    
    l = loss(y_pred, y_train)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss={l.item():.4}")

with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.round()  # this is why we use torch.no_grad to prevent this from being part of computational graph
    acc = y_pred.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy={acc:.4}")
