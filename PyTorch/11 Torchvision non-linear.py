from multiprocessing.connection import deliver_challenge
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
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

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Length of training data: {len(train_data)}\nLength of testing data: {len(test_data)}")
class_names = train_data.classes
class_to_idx = train_data.class_to_idx
train_dataloader = torch.utils.data.DataLoader(dataset = train_data, 
                                               batch_size = BATCH_SIZE,
                                               shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_data,
                                              batch_size = BATCH_SIZE,
                                              shuffle = False)

print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch, train_labels_batch = train_features_batch.to(device), train_labels_batch.to(device)
def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct / len(y_preds) * 100)
    return acc

def print_train_time(start: float,
                     end: float, 
                     device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f}")
    return total_time

class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_features: int,
                 hidden_units: int,
                 output_features: int):
        super().__init__()
        self.stack_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_features,
                      out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units,
                      out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units,
                      out_features = output_features)
        )
    def forward(self, x: torch.Tensor):
        return self.stack_layer(x)

model_1 = FashionMNISTModelV1(input_features = 28 * 28,
                              hidden_units = 10,
                              output_features = len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_1.parameters(),
                            lr = 0.1)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model_1.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        preds = model_1(X)
        loss = loss_fn(preds, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true = y, y_preds = preds.argmax(dim = 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 96 == 0:
            print(f"Looked at {batch * len(X)} / {(len(data_loader.dataset))} samples.", end = "\r")
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

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
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss += loss_fn(preds, y)
            acc += accuracy_fn(y_true = y, y_preds = preds.argmax(dim = 1))
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"Model Name": model.__class__.__name__,
            "Model loss": loss.item(),
            "model_acc": acc}

epochs = 3
train_time_start_on_gpu = timer()
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n------")
    train_step(model = model_1,
               data_loader = train_dataloader,
               loss_fn = loss_fn,
               optimizer = optimizer,
               accuracy_fn = accuracy_fn,
               device = device)
    test_step(model = model_1,
              data_loader = test_dataloader,
              loss_fn = loss_fn,
              accuracy_fn = accuracy_fn,
              device = device)
train_time_end_on_gpu = timer()
total_train_time = print_train_time(start = train_time_start_on_gpu, end = train_time_end_on_gpu, device = device)
model_1_results = eval_model(model = model_1,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn,
                             device = device)
print(model_1_results)

