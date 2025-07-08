import torch
from torchvision import transforms
from PIL import Image
from Model import LeNet
import matplotlib.pyplot as plt


def detect (image_path , model_path = 'LeNet_Fashion_MNIST.pth'):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # batch → [1, 1, 28, 28]

    model = LeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        label = predicted.item()

        label_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        print("Predicted Label:", label_names[label])
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Predicted Label: {label_names[label]}')
        plt.axis('off')
        plt.show()


def main():
    detect('data_for_test/Sneaker.jpg','LeNet_Fashion_MNIST.pth')

if __name__ == '__main__':
    main()