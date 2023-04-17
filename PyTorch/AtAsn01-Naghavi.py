import torch
import torchvision
import torchvision.datasets as datasets
import random

# Load CIFAR-10 dataset
train = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# Subset 100 images from each class
subset_count = 100
classes = train.classes
subset_indices = []
for target_class in range(len(classes)):
    class_indices = torch.nonzero(torch.tensor(train.targets) == target_class).flatten().tolist()
    subset_indices += class_indices[:subset_count]

subset = torch.utils.data.Subset(train, subset_indices)


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
x, y = subset[random.randint(0, len(subset))]
conv = torchvision.transforms.ToTensor()
x = torch.unsqueeze(conv(x), 0).reshape((1, 3072)) # Add batch dimension

hinge_predictions = torch.matmul(x, w)
hinge_loss_val = hinge_loss(hinge_predictions, torch.tensor([y]))

softmax_predictions = softmax(x, w)
softmax_loss = -torch.log(softmax_predictions[:, y])
softmax_loss_val = torch.nn.functional.cross_entropy(softmax_predictions, torch.tensor([y]))

print(f'Hinge loss: {hinge_loss_val:.4f}, Softmax loss: {softmax_loss_val:.4f}  {softmax_loss.item()}, Selected picture: {classes[y]}')
