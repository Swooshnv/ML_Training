import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load CIFAR-10 dataset
train = datasets.CIFAR10(root='./data', train=True, download=False, transform=ToTensor())
def sub_set(data):
    # Subset 100 images from each class
    subset_count = 100
    classes = data.classes
    subset_indices = []
    for target_class in range(len(classes)):
        class_indices = torch.nonzero(torch.tensor(train.targets) == target_class).flatten().tolist()
        subset_indices += class_indices[:subset_count]
    subset = torch.utils.data.Subset(data, subset_indices)
    return subset

X = sub_set(train)
b1 = np.random.random(size = (1,1500))
W1 = np.random.random(size = (3072,1500))
b2 = np.random.random(size = (1,1000))
W2 = np.random.random(size = (1500,1000))
b3 = np.random.random(size = (1,10))
W3 = np.random.random(size = (1000,10))

#f = lambda y: np.maximum(0, y)    #ReLU
#f = lambda y: np.tanh(y)    #Tanh
f = lambda y: 1.0 / (1.0 + np.exp(-y))    #Sigmoid
x = np.reshape(X[0][0], (3072))
h1 = f(np.dot(x, W1) + b1)
h2 = f(np.dot(h1, W2) +b2)
out = np.dot(h2, W3) + b3
print(f"First forward pass scores: {out}")