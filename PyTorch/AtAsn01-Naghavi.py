import torch
import torchvision
import torchvision.datasets as datasets
import random

# Load CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# Subset 100 images from each class
num_images_per_class = 100
classes = cifar10_train.classes
subset_indices = []
for target_class in range(len(classes)):
    class_indices = torch.nonzero(torch.tensor(cifar10_train.targets) == target_class).flatten().tolist()
    subset_indices += class_indices[:num_images_per_class]

cifar10_subset = torch.utils.data.Subset(cifar10_train, subset_indices)


def hinge_loss(predictions, targets):
    margin = 1
    loss = torch.max(torch.zeros_like(predictions), margin - predictions * targets)
    return torch.mean(loss)

def softmax(x, w):
    logits = torch.matmul(x, w)
    exp_logits = torch.exp(logits)
    probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
    return probs



# Generate random weight matrix with shape (3072, 10)
num_features = 3072
num_classes = len(classes)
w = torch.randn(num_features, num_classes)

# Classify a random image using hinge loss and softmax
x, y = cifar10_subset[random.randint(0, len(cifar10_subset))]
conv = torchvision.transforms.ToTensor()
x = torch.unsqueeze(conv(x), 0).reshape((1, 3072)) # Add batch dimension

hinge_predictions = torch.matmul(x, w)
hinge_loss_val = hinge_loss(hinge_predictions, torch.tensor([y]))

softmax_predictions = softmax(x, w)
softmax_loss = -torch.log(softmax_predictions[:, y])
softmax_loss_val = torch.nn.functional.cross_entropy(softmax_predictions, torch.tensor([y]))

print(f'Hinge loss: {hinge_loss_val:.4f}, Softmax loss: {softmax_loss_val:.4f}  {softmax_loss.item()}, Selected picture: {cifar10_train.classes[y]}')