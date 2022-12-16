import random
import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

BATCH_SIZE = 32
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True,
                 exist_ok = True)
MODEL_NAME = "12_MODEL.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
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

LOADED_MODEL = FashionMNISTCNNV0(input_shape = 1,
                            hidden_units = 10,
                            output_shape = len(class_names)).to(device)
LOADED_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH))                                    
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)
    
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
    print(f"Test sample label: {test_labels[-1]} ({class_names[test_labels[-1]]})")

pred_probs= make_predictions(model=LOADED_MODEL, 
                             data=test_samples)
for i in range(9):
    print(f" The input image is classified as: {class_names[pred_probs[i].argmax()]}")
