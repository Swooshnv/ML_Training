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
X_train, y_train, X_test, y_test = train_test_split(X, 
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
print(next(model_0.parameters()).device)

### Replicate model using nn.sequential
model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5),
                        nn.Linear(in_features=5, out_features=1)).to(device)
print(f"Replica: {next(model_0.parameters()).device}")

### Setting up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
