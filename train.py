from dataset import TeamAssignmentDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2

class EmbeddingCNN(nn.Module):
    """
    Expects 120x120 images
    """
    def __init__(self):
        self.input_size = 120 # assume square input
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (self.input_size // (2**3)) * 2, 512)  # 120x120 each dimension divided by 2 for each max pooling
        self.fc2 = nn.Linear(512, 1024)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, 64 * (self.input_size // (2**3)) * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def process_image(img):
    # Convert grayscale to RGB if necessary
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Convert to PyTorch tensor, rearrange channels, and normalize
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img

def collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    # Process and stack all images
    anchors = torch.stack([process_image(img) for img in anchors])
    positives = torch.stack([process_image(img) for img in positives])
    negatives = torch.stack([process_image(img) for img in negatives])

    return anchors, positives, negatives

train_dataset = TeamAssignmentDataset(is_train_set=True)
test_dataset = TeamAssignmentDataset(is_train_set=False)

train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

model = EmbeddingCNN()

# Loss and optimizer
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):
    for data in train_dataloader:
        anchor, positive, negative = data

        # Forward pass for each batch
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")

    # Evaluate on test set after each epoch
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    with torch.no_grad():  # No gradients needed for evaluation
        for data in test_dataloader:
            anchor, positive, negative = data

            # Forward pass for each batch
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            test_loss = criterion(anchor_out, positive_out, negative_out)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Test Loss: {avg_test_loss}")