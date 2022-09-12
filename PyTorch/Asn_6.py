import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

### Create test data
weight = 1.3
bias = 0.2
start = 1
stop = 5
step = 0.1

X = torch.arange(start, stop, step).unsqueeze(dim = 1)
y = (weight * X) + bias
print(X)
print(y)

### Creating a train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y)
print(len(X_train), len(X_test), len(y_train), len(y_test))

### Visualizing data

### PyTorch: Building a linear regression model
class linearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

### Creating a loss function and an optimizer
torch.manual_seed(42)
model0 = linearRegressionModel()
print(list(model0.parameters()))
loss_fn = nn.L1Loss()

with torch.inference_mode():
    y_preds = model0(X_test)
    print(y_preds)
    
    # Visualize
    """
    plt.scatter(X_train, y_train, c="b", s=4, label = "Training Data")
    plt.scatter(X_test, y_test, c="r", s=4, label = "Testing Data")
    plt.scatter(X_test, y_preds, c="g", s=4, label = "Predicted Data")
    plt.legend()
    plt.show()
    """
optimizer = torch.optim.SGD(params = model0.parameters, lr=0.01)

### Creating a training loop and a testing loop
epochs = 1
for epoch in epochs:
    model0.train()
    y_pred = model0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model0.eval()
