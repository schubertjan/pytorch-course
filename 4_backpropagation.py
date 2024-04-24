"""https://youtu.be/c36lUUr864M?feature=shared&t=2520"""
import torch

x = torch.tensor(1)
y = torch.tensor(2)
w = torch.tensor(1.0, requires_grad=True)

# formward pass
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

# backward pass
loss.backward()
print(w.grad)
