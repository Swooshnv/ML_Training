import torch
import torchvision
import torchvision.datasets as datasets
import random

# Load CIFAR-10 dataset
train = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

def sub_set(data):
    # Subset 100 images from each class
    subset_count = 100
    classes = data.classes
    subset_indices = []
    for target_class in range(len(classes)):
        class_indices = torch.nonzero(torch.tensor(train.targets) == target_class).flatten().tolist()
        subset_indices += class_indices[:subset_count]
    subset = torch.utils.data.Subset(train, subset_indices)
    return subset

def hinge_loss(predictions, targets):
    margin = 1
    loss = torch.max(torch.zeros_like(predictions), margin - predictions * targets)
    print(f"Hinge Loss: {torch.mean(loss)}")
    return torch.mean(loss)

def softmax(x, w, y):
    logits = torch.matmul(x, w)
    exp_logits = torch.exp(logits)
    probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
    softmax_loss = -torch.log(probs[:, y])
    softmax_loss_val = torch.nn.functional.cross_entropy(probs, torch.tensor([y]))
    print(f"Sofmax: {softmax_loss_val})")

def calculate(subset, data):
    # Generate random weight matrix with shape (3072, 10)
    classes = data.classes
    num_features = 3072
    num_classes = len(classes)
    w = torch.randn(num_features, num_classes)

    # Classify a random image using hinge loss and softmax
    x, y = subset[random.randint(0, len(subset))]
    conv = torchvision.transforms.ToTensor()
    x = torch.unsqueeze(conv(x), 0).reshape((1, 3072)) # Add batch dimension
    hinge_predictions = torch.matmul(x, w)
    softmax(x, w, y)
    hinge_loss(hinge_predictions, torch.tensor([y]))

subset = sub_set(train)
calculate(subset, train)
