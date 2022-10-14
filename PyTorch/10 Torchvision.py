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
                                              shuffle = False)

print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))

### Creating a baseline model
flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)
print(f"shape before flattening: {x.shape}")
print(f"shape after flattening: {output.shape}")

def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct / len(y_preds) * 100)
    return acc

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_features : int,
                 output_features : int,
                 hidden_units : int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_features,
                      out_features = hidden_units),
            nn.Linear(in_features = hidden_units,
                      out_features = output_features)
        )
    def forward(self, x):
        return self.layer_stack(x)

model_0 = FashionMNISTModelV0(
    input_features = 28 * 28,
    hidden_units = 10,
    output_features = len(class_names)).to("cpu")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.1)

### Creating a timer
def print_train_time(start: float,
                     end: float, 
                     device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f}")
    return total_time

### Creating a training/testing loop with batches of data
train_time_start_on_cpu = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n-----")
    train_loss = 0
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
            test_acc = accuracy_fn(y_true = y_test, y_preds = test_preds.argmax(dim = 1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start = train_time_start_on_cpu,
                                            end = train_time_end_on_cpu,
                                            device = str(next(model_0.parameters()).device))

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            preds = model(X)
            loss += loss_fn(preds, y)
            acc += accuracy_fn(y_true = y, y_preds = preds.argmax(dim = 1))
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"Model Name": model.__class__.__name__,
            "Model loss": loss.item(),
            "model_acc": acc}

model_0_results = eval_model(model = model_0,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn)
print(model_0_results)

