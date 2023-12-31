from dataset import TeamAssignmentDataset
from embedding_model import EmbeddingCNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2

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

NUM_EPOCHS = 2

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

weights_dir = '/Users/daniel/Desktop/Soccer-Team-Assignment/weights/model_weights.pth'
torch.save(model.state_dict(), weights_dir)
print('Weights saved to dir: ', weights_dir)