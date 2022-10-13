import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print(torch.__version__)
print(torchvision.__version__)

### Downloading Dataset
train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = False,
    transform = ToTensor(),
    target_transform = None,

)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = False, 
    transform = ToTensor(),
    target_transform = None
)
print(f"Length of training data: {len(train_data)}\nLength of testing data: {len(test_data)}")
class_names = train_data.classes
class_to_idx = train_data.class_to_idx

image, label = train_data[0]
plt_image = image.squeeze()
plt.imshow(plt_image, cmap = "gray")
plt.title(class_names[label])
plt.show()