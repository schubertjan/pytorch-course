"""https://youtu.be/c36lUUr864M?feature=shared&t=4373"""

import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32).reshape(-1, 1)  # 2D number of rows = number of samples
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32).reshape(-1, 1)  # 2D number of rows = number of samples

# Define model
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)  # use just one layer

# Or when designing a model
class LinearRegresion(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegresion, self).__init__()
        # define layers
        self.lin = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegresion(input_size, output_size)

X_test = torch.tensor([5.0], dtype=torch.float32)
print(f"Prediction before training: f(5) {model(X_test).item():.3f}")

# Training
lr = 0.001
n_iters = 200
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    # forward
    y_pred = model(X)
    # loss
    l = loss(y, y_pred)
    # gradient
    l.backward()
    # update weights
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        w, b = model.parameters()
        print(f"epoch {epoch + 1}: w={w.item():.3f}, loss={l:.8f}") 

print(f"Prediction after training: f(5) {model(X_test).item():.3f}")
