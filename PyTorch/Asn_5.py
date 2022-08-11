import torch
import numpy as np

### PyTorch: Indexing tensors
'''
tensor = torch.arange(1, 10).reshape(1, 3, 3)
print(f"Shape of the tensor: {tensor.shape}")
print(f"Tensor[0] --> {tensor[0]}\n")
print(f"Tensor[0, 1] --> {tensor[0, 1]}\n")
print(f"Tensor[0, 1, 2] --> {tensor[0, 1, 2]}\n")

#Get the third input from all second dimensions
print(f"Third inputs from second dimensions: {tensor[:, :, 2]}")
'''

### PyTorch: Interacting with numpy arrays
'''
array = np.arange(1.0, 10.0)
tensor = torch.from_numpy(array)
print(f"Numpy array: {array, array.dtype}\nTorch converted tensor: {tensor, tensor.dtype}")
tensor0 = torch.zeros(size = (2, 5))
torch_array = tensor0.numpy()
print(f"Tensor: {tensor0, tensor0.dtype}\nTensor (converted to numpy): {torch_array, torch_array.dtype}")
'''

### PyTorch: Reproducability
'''
SEED = 2048
torch.manual_seed(SEED)
random0 = torch.rand(size = (2, 4))
torch.manual_seed(SEED)
random1 = torch.rand(size = (2, 4))
print(f"First tensor: {random0}\nSecond tensor: {random1}\nAre both tensors equal? {random0 == random1}")
'''

### PyTorch: Interaction with GPU
'''
print(f"Is CUDA available on your system? {torch.cuda.is_available()}")
print(f"How many CUDA devices are available? {torch.cuda.device_count()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = torch.arange(1., 11.)
tensor_gpu = tensor.to(device)
print(f"Default tensor device: {tensor.device}\nConverted tensor device: {tensor_gpu.device}")
tensor_revert= tensor_gpu.cpu().numpy()
print(f"Reverted tensor to np array: {tensor_revert}")
'''