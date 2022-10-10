from typing_extensions import Self
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

c_samples = 1000
X, y = make_circles(c_samples, 
                    noise = 0.03,
                    random_state = 42)

circles = pd.DataFrame({"X1": X[: , 0],
                       "X2": X[: , 1], 
                       "label": y})
plt.scatter(x = X[: , 0],
            y = X[: , 1],
            c = y,
            cmap = plt.cm.RdYlBu)
#plt.show()            
#print(type(X))

### Turning created data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

### Train test splitting the tensors
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 42)

### Building a model
device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

model_0 = CircleModelV1().to(device)
#print(next(model_0.parameters()).device)

### Replicate model using nn.sequential
"""
model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5),
                        nn.Linear(in_features=5, out_features=1)).to(device)
print(f"Replica: {next(model_0.parameters()).device}")
"""

### Setting up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.1)

def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct / len(y_preds) * 100)
    return acc

### Test data evaluation
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
print(y_test[:5])

y_pred_probs = torch.sigmoid(y_logits)
print(f"Sigmoid activation: {y_pred_probs}")
print(f"Rounded outputs:{torch.round(y_pred_probs)}")

### Building training/testing loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 100
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    acc = accuracy_fn(y_true = y_train,
                      y_preds = y_pred)
    loss = loss_fn(y_logits,
                   y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        y_test_preds = model_0(X_test).squeeze()
        rounded = torch.round(torch.sigmoid(y_test_preds))
        
        test_loss = loss_fn(y_test_preds,
                    y_test)
        test_acc = accuracy_fn(y_true = y_test,
                        y_preds = rounded)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | loss: {loss:.5f} | acc: {acc:.5f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}%")

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV2().to(device)

loss_fn_1 = nn.BCEWithLogitsLoss()
optimizer_1 = torch.optim.SGD(params = model_1.parameters(),
                              lr = 0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print("***** Model_1 Training/testing loop *****")
for epoch in range(epochs):
    model_1.train()
    y_preds_1 = model_1(X_train).squeeze()
    y_round = torch.round(torch.sigmoid(y_preds_1))
    loss_1 = loss_fn_1(y_round, y_train)
    acc = accuracy_fn(y_true = y_train,
                      y_preds = y_round)
    optimizer_1.zero_grad()
    loss_1.backward()
    optimizer_1.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_round = torch.round(torch.sigmoid(test_logits))
        test_loss_1 = loss_fn_1(test_round, y_test)
        test_acc_1 = accuracy_fn(y_true = y_test, y_preds = test_round)
    
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | loss: {loss_1:.5f} | accuracy: {acc:.2f}% | test loss: {test_loss_1:.5f} | test accuracy: {test_acc_1:.2f}%")
