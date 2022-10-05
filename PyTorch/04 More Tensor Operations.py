import torch

### PyTorch: Manipulating tensors
'''
a = torch.arange(0., 20.)
a_resized = a.resize(4, 5)
a_resized2 = a.resize(20, 1)
print(f"Original tensor: {a}\nResized tensor: {a_resized}\nSecond Resized tensor: {a_resized2}")

tensor_view = a.view(5, 4)
tensor_view[0:3, 0:1] = 5.
print(f"Resulting tensor: {a}")
'''

'''
first = torch.tensor([1., 2., 3.])
second = torch.tensor([4., 5., 6.])
h_stacked = torch.hstack([first, second])
v_stacked = torch.vstack([first, second])
print(f"H stacked tensor: {h_stacked}\nV stacked tensor: {v_stacked}")
stacked0 = torch.stack([first, second], dim = 0)
stacked1 = torch.stack([first, second], dim = 1)
print(f"Dim0 stacked: {stacked0}\nDim1 stacked: {stacked1}")
'''
#squeezing/unsqueezing

test_tensor = torch.rand(size = (2, 5))
unsqueezed = test_tensor.unsqueeze(dim = 0)
print(f"Original tensor: {test_tensor, test_tensor.shape}\nUnsqueezed tensor: {unsqueezed, unsqueezed.shape}")
squeezed = unsqueezed.squeeze()
print(f"Squeezed tensor: {squeezed, squeezed.shape}")

# tensor permutation

tensor0 = torch.rand(size = (2, 4, 3))
tensor0_permuted = tensor0.permute(dims = (2, 0, 1))
print(f"\nTensor before permutation: {tensor0, tensor0.shape}\nTensor after permutation: {tensor0_permuted, tensor0_permuted.shape}")
