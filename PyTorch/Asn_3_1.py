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
BATCH_SIZE = 32

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

### Plotting one image
"""
image, label = train_data[0]
plt_image = image.squeeze()
plt.imshow(plt_image, cmap = "gray")
plt.title(class_names[label])
plt.show()
"""
### Plotting multiple images at once
"""
fig = plt.figure(figsize = (9, 9))
rows, cols = 4, 4 
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size = [1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap = "gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.show()
"""
train_dataloader = torch.utils.data.DataLoader(dataset = train_data, 
                                               batch_size = BATCH_SIZE,
                                               shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_data,
                                              batch_size = BATCH_SIZE,
                                              shuffle = True)
