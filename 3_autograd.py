"""https://youtu.be/c36lUUr864M?feature=shared&t=1561"""

import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 2
print(z)

z = z.mean()
print(z)

# calculate gradient of x
z.backward()  # dz/dx; must be a scaler
print(x.grad)


v = torch.tensor([1., 2., 3.])
z = y * y * 2
z.backward(v)  # dz/dx if z not scaler
print(z)
print(x.grad)

# REmove gradient tracking 
x.requires_grad_(False)
print(x)

y = x.detach() # will require a new vector with no gradient
print(y)

with torch.no_grad():
    y = x + 2
    print(y)


# grad attributes is cummulative
weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)

    weights.grad.zero_()  # cleans the gradience
    print(weights.grad)
