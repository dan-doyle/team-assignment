from dataset import TeamAssignmentDataset
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50

# TODO: initialise dataset
# dataset = TeamAssignmentDataset(...)

# TODO: DataLoader setup
# dataloader = DataLoader(dataset, batch_size=..., shuffle=True)

# Model setup (using ResNet50 for feature extraction as an example)
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 128)  # Adjust output features

# Loss and optimizer
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        anchor, positive, negative = data  # Your dataloader should return triplets
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
