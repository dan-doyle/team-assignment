import torch.nn as nn
import torch.nn.functional as F

class EmbeddingCNN(nn.Module):
    """
    Expects 120x120, 3 channel images
    """
    def __init__(self):
        self.input_size = 120 # assume square input
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * ((self.input_size // (2**3)) ** 2), 512)  # 120x120 each dimension divided by 2 for each max pooling
        self.fc2 = nn.Linear(512, 1024)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, 64 * ((self.input_size // (2**3)) ** 2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x