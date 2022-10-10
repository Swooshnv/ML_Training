import torch
import matplotlib.pyplot as plt
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
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
plt.show()            

X_train, y_train, X_test, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_0 = nn.Sequential(
    nn.Linear(in_features = 2, out_features = 10)
    nn.Linear(in_features = 10, out_features = 10)
    nn.Linear(in_features = 10, out_features = 1).to(device)
)

