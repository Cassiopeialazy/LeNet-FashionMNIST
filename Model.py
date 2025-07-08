import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),  # (1, 28, 28) → (6, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # → (6, 14, 14)

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # → (16, 10, 10)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # → (16, 5, 5)

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),  # → (120, 1, 1)
            nn.ReLU(),

            nn.Flatten(),  # → (120)

            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)



