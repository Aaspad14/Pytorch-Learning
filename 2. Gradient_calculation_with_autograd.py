from enum import Flag
from numpy import float32
import torch

x = torch.randn(3, requires_grad=True)  #Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
print(x)

y = x+2
print(y)

z = y*y*2
print(z)

z = z.mean()
print(z)
z.backward()    #Backword() function can be used only on scalar values.

print(x.grad)

z = y*y*2
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) #dz/dx if z is not scalar then we can pass a vector as argument.
print(x.grad)

# ************ To remove the gradient from a tensor ****************

# x.requires_grad_(False)
print(x)
x.requires_grad_(False)
print(x)  # With grad removed

# x.detach()
print(x)
print(x.detach())

# with torch.no_grad():
with torch.no_grad():
    y = x + 2
    print(y)
    

# ************ Whenever we call the .backward() function, the gradient for the tensor gets accumulated into the .grad attribute so their values get summed up **********
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad) # For multiple iterations the gradient for the tensor gets accumulated into the .grad attribute

    # weights.grad.zero_() # To empty the gradients after each iteration.



# *************** Built-in optimizer for Py-torch ***************************
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()