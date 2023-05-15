import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
import random
import sklearn
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

# Load CIFAR-10 dataset
train = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
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
def test_sub_set(data):
    # Subset 100 images from each class
    subset_count = 25
    classes = data.classes
    subset_indices = []
    for target_class in range(len(classes)):
        class_indices = torch.nonzero(torch.tensor(train.targets) == target_class).flatten().tolist()
        subset_indices += class_indices[:subset_count]
    subset = torch.utils.data.Subset(data, subset_indices)
    return subset

class atNN(nn.Module):
    def __init__(self,
                 input_features : int,
                 hidden_units : int,
                 hidden_units_two : int,
                 output_features : int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_features,
                      out_features = hidden_units),
            nn.Sigmoid(),
            nn.Linear(in_features = hidden_units,
                      out_features = hidden_units_two),
            nn.Sigmoid(),
            nn.Linear(in_features = hidden_units_two,
                      out_features = output_features)
        )
    def forward(self, x):
        return self.layer_stack(x)

model_0 = atNN(input_features= 3072,
               hidden_units = 7500,
               hidden_units_two = 3000,
               output_features = 10)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.1)

X = sub_set(train)
X_test = test_sub_set(test)
#X = transform(X)
train_dataloader = torch.utils.data.DataLoader(dataset = X, 
                                               batch_size = 32,
                                               shuffle = False)
test_dataloader = torch.utils.data.DataLoader(dataset = X_test,
                                              batch_size = 32,
                                              shuffle = False)

epochs = 5
for epoch in tqdm(range(epochs)):
    train_loss = 0
    print(f"\nEpoch: {epoch}\n-----")
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        preds = model_0(X)
        loss = loss_fn(preds, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 96 == 0:
            print(f"Looked at {batch * len(X)} / {(len(train_dataloader.dataset))} samples.", end = "\r")    
    train_loss /= len(train_dataloader)
    test_loss, test_acc = 0, 0
    
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_preds = model_0(X_test)
            test_loss += loss_fn(test_preds, y_test)
        test_loss /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")

