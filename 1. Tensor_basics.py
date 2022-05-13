import torch
import numpy as np

x = torch.empty(1) #1d tensor
x = torch.empty(2, 3)  #2d tensor
x = torch.rand(2, 2) #2d tensor with random variables
x = torch.zeros(2,2) #tensor with all elements as 0.
x = torch.ones(2,2) #tensor with all elements as 1.
print(x)
x = torch.ones(2,2, dtype=int) # or double/float16 to change the dtype
print(x.dtype) # default datatype of a tensor element is float32
x = torch.tensor([2.5,0.1]) #To create a tensor from a list.
print(x.size()) #to check the size of a tensor.



#*********************Operations on tensor************************

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x+y     #To add the corresponding elements of 2 tensors.
z = torch.add(x,y) # we can also use the add() function to achieve the same.

y.add_(x) # In pytorch every function with a trailing underscoree implies a inplace operation.
print(y)

#Substraction
z = x - y
z = torch.sub(x, y)
print(z)

#Multiplication
z = x * y
z = torch.mul(x, y)
print(z)
y.mul_(x)
print(y)

#Multiplication
z = x / y
z = torch.div(x, y)
print(z)
y.div_(x)
print(y)

#Slicing on tensors
x = torch.rand(5, 3)
print(x)
print(x[:,0]) #All rows and only first column
print(x[1,1]) #Element of 2nd row and 2nd column in a tensor
print(x[1,1].item()) #gives the element of 2nd row and 2nd column as an item.


#Resizing on tensors
x = torch.rand(4,4)
print(x)
y = x.view(-1, 8)  # -1 bydefault determines the correseponding rows for the columns mentioned.
print(y.size())


#Convert a tensor to a numpy array
a = torch.ones(5) #tensor of all ele ones
print(a)
b = a.numpy() #a numpy array of same elements as the tensor 
print(type(b))

a.add_(1)
print(a)
print(b)  #Both tensor and numpy array gets updated as they both point to the same memory location.

#Convert a numpy array to tensor
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)

# To perform task on GPU instead of CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu") # a gpu tensor can't be converted into a numpy array so we need to convert it to cpu.

# To optimise a variable later we give requires_grad=True
x = torch.ones(5, requires_grad=True)
print(x)