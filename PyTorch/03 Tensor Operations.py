import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random_tensor = torch.rand(4, 4)
zeroes_tensor = torch.zeros(4, 4)
ones_tensor = torch.ones(size = (4, 4))
ranged_tensor = torch.arange(start = 10,end = 15,step = 0.5)
zeroes_like = torch.zeros_like(input = ranged_tensor)
#print(ranged_tensor)
#print(random_tensor * zeroes_tensor)
#print(zeroes_like)

### PyTorch: interacting with tensors and attributes
'''
cuda = torch.device('cuda')
float_32 = torch.arange(0, 5, 0.3, dtype = torch.float32, device = cuda, requires_grad = False)
print(f"Data type of this tensor is: {float_32.dtype}")
print(f"This tensor is made on: {float_32.device}")
float_16 = float_32.type(torch.float16)
print(float_16.dtype)
'''

### PyTorch: Element-wise multiplication and Matrix multiplication
'''
ten1 = torch.arange(1, 5, dtype = torch.int, device = None, requires_grad = False)
ten2 = torch.arange(5, 9, dtype = torch.int, device = None, requires_grad = False)
print(f"ten1 = {ten1}")
print(f"ten2 = {ten2}")
print(f"ten1 * ten2 = {ten1 * ten2}")
matmul = torch.matmul(ten1, ten2)
print(f"Matrix multiplication of ten1 and ten2: {matmul}")
'''

### PyTorch: Tensor transposition, multiplication and aggregation
'''
tensor_one = torch.tensor([[1, 2],
                           [3, 4],
                           [5, 6]])
tensor_two = torch.tensor([[3, 2],
                           [7, 14],
                           [21, 10]])
tensor_prod = torch.matmul(tensor_one, tensor_two.T)
print(f"Shape of tensor_one is: {tensor_one.shape}")
print(f"Shape of tensor_two is: {tensor_two.shape}")
print(f"Shape of trasposed tensor_two is: {tensor_two.T.shape}")
print(f"Dot product of these two tensors is: {tensor_prod}")

print(f"\nMinimum of Dot prod is: {torch.min(tensor_prod)}")
print(f"Mean of Dot prod is: {torch.mean(tensor_prod.type(torch.float32))}")
print(f"Sum of Dot prod is: {torch.sum(tensor_prod)}")
'''

### PyTorch: Locating min and max of a tensor
tensor = torch.arange(0, 150, step = 10, dtype = torch.int32)
print(tensor)
print(tensor.argmin())
print(tensor.argmax())
print(tensor[tensor.argmax()].item())
