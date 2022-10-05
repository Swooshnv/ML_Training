import torch
import numpy as np
import pandas as pf
import matplotlib.pyplot as plt
print(torch.__version__)

#Scalar
Scalar = torch.tensor(7)
print(Scalar)

print("Scalar ndim: {}".format(Scalar.ndim))
print("Scalar shape: {}".format(Scalar.shape))

#Vector
vector = torch.tensor([3,5])
print(vector)

print("vector ndim: {}".format(vector.ndim))
print("vector shape: {}".format(vector.shape))

#MATRIX
MATRIX = torch.tensor([[5,6],
                       [7,8]])
print(MATRIX)

print("matrix ndim: {}".format(MATRIX.ndim))
print("matrix shape: {}".format(MATRIX.shape))
print(MATRIX[1][0])

#TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
print(TENSOR)

print("tensor ndim: ".format(TENSOR.ndim))
print("tensor shape: ".format(TENSOR.shape))
print(TENSOR[0])
print(TENSOR[0][1][2])
