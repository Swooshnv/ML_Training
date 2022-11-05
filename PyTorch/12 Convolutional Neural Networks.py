import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from tqdm.auto import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

BATCH_SIZE = 32

train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = False,
    transform = ToTensor(),
    target_transform = None
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = False,
    transform = ToTensor(),
    target_transform = None
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Length of training data: {len(train_data)}\nLength of testing data: {len(test_data)}")
class_names = train_data.classes
class_to_idx = train_data.class_to_idx
train_dataloader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = False
)

print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch, train_labels_batch = train_features_batch.to(device), train_labels_batch.to(device)

def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_preds, y_true).sum().item()
    return (correct/len(y_preds))*100

def print_time(start:float, end:float, device: torch.device = None):
    total_time = end - start
    print(f"Total time on {device}: {total_time:.3f}")
    return total_time

### Creating my first convolutional neural network :)
class FashionMNISTCNNV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = input_shape,
                out_channels = hidden_units,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2
            )
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2
            )
        )
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features = hidden_units * 7 * 7,
                out_features = output_shape
            )
        )
    def forward(self, x):
        x = self.conv_block_1(x)
        #print(f"Output shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        #print(f"Output shape of conv_block_2: {x.shape}")
        x = self.classifer(x)
        #print(f"Output shape of classifier: {x.shape}")
        return x

model_0 = FashionMNISTCNNV0(input_shape = 1,
                            hidden_units = 10,
                            output_shape = len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.1)

def train_step(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn,
                device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true = y, y_preds = preds.argmax(dim = 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 96 == 0:
            print(f"Looked at {batch * len(X)} / {len(data_loader.dataset)} samples.", end = "\r")
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f}\nTrain accuracy: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            test_loss += loss_fn(preds, y)
            test_acc += accuracy_fn(y_true = y, y_preds = preds.argmax(dim = 1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f}\nTest accuracy: {test_acc:.2f}%")
        
epochs = 5
train_time_start = timer()
for epoch in tqdm(range(epochs)):    
    print(f"\nEpoch: {epoch}\n---------")
    train_step(model = model_0, 
                data_loader = train_dataloader,
                loss_fn = loss_fn,
                optimizer = optimizer,
                accuracy_fn = accuracy_fn,
                device = device)

    test_step(model = model_0,
            data_loader = test_dataloader,
            loss_fn = loss_fn,
            accuracy_fn = accuracy_fn,
            device = device)

train_time_end = timer()
total_train_time = print_time(start = train_time_start,
                              end = train_time_end,
                              device = device)
