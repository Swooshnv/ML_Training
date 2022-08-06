from random import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random_tensor = torch.rand(2, 3, 3)
print(random_tensor)

print("random_tensor ndim: {}".format(random_tensor.ndim))
print("random_tensor shape: {}".format(random_tensor.shape))

#Image tensor
random_image = torch.rand(size = (256, 256, 3))

print("random_image ndim: {}".format(random_image.ndim))
print("random_image shape: {}".format(random_image.shape))