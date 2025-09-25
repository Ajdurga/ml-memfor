import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding =1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3,padding = 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding =1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        with torch.no_grad():
            d = torch.zeros(1,3,32,32)
            d = self.block1(d); d = self.block2(d)
            flat = d.flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
