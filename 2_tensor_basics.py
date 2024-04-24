"""https://youtu.be/c36lUUr864M?feature=shared&t=450"""

import torch
import numpy as np

x = torch.empty(2, 3)
print(x)

x = torch.rand(2, 3)
print(x)

x = torch.ones(2, 3)
print(x)

x = torch.zeros(2, 3)
print(x)

x = torch.empty(2, 3, dtype=torch.int)
print(x)

x = torch.ones(2, 3, 3)
print(x.size())


# operations
x = torch.ones(2, 3)
y = torch.ones(2, 3)
z = x * y
z = torch.mul(x, y) # same operation
print(z)

y.mul_(x) # inplace adding
print(y)

# indexing
x = torch.rand(5, 3)
print(x[:, 0])
print(x[:2, 0])

# reshaping
x = torch.rand(4, 3)
print(x.view(-1, 1))
print(x.view(2, 6))

# data types
print(x.numpy())
print(torch.from_numpy(np.zeros(1)))


# on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    z = torch.ones(5, device=device)
    z = x.to(device=device)


# create a variable tensor
x = torch.ones(5, requires_grad=True)








