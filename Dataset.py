import torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.FashionMNIST(root='./dataset', train=True, transform=transform, download=False)
test_set = torchvision.datasets.FashionMNIST(root='./dataset', train=False, transform=transform, download=False)