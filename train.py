from dataset import TeamAssignmentDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class EmbeddingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 15 * 15, 512)  # Adjust dimensions according to your pooling layers
        self.fc2 = nn.Linear(512, 1024)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, 64 * 15 * 15)  # Adjust dimensions according to your pooling layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

dataset = TeamAssignmentDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = EmbeddingCNN()

# Loss and optimizer
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 20

# Training loop
for epoch in range(NUM_EPOCHS):
    for data in dataloader:
        anchor, positive, negative = data  # Your dataloader should return triplets
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")