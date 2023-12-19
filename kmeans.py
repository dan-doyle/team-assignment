import json
import cv2
import torch
from sklearn.cluster import KMeans
from dataset import TeamAssignmentDataset
from embedding_model import EmbeddingCNN

"""
- set up dataset class
- access frame from video using class
- method: display frame with bounding boxes drawn
- run K-means
"""

class DisplayFrame():
    def __init__(self, dataset, frame_number):
        """
        - Take an index from the JSON e.g. 20 !
        - Fetch the frame from the video
        - Plot bboxes
        """
        self.dataset = dataset
        self.frame_number = str(frame_number)
        """
        For best software engineering practice, we should receive the video path, frame_number, bboxes. But here we couple the Dataset class with this one
        """
        with open(self.dataset.label_dir) as json_file:
            data = json.load(json_file)
        self.player_coords = data['frame_data'][self.frame_number] # dict with key: player_id and value: xtl, xbr, ytl, ybr
        self.image = self.grab_img()

    def grab_img(self):
        cap = cv2.VideoCapture(self.dataset.video_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.frame_number))
        ret, frame_img = cap.read() # read the current frame
        cap.release()
        return frame_img

    def show_img(self, teams):
        """
        teams of format: {0: set(), 1: set()}
        """
        img = self.image.copy()
        for player_id, value in self.player_coords.items():
            coords = value['coords']
            top_left = (coords[0], coords[1])
            bottom_right = (coords[2], coords[3])
            if player_id not in teams: # Player not annotated? TODO: we need to pass every player for inference, but currently not all players are stored in frames
                continue
            if teams[player_id] == 0:
                img = cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)  # Red 
            else:
                img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2) # Yellow
        cv2.imshow('Image', img)
        cv2.waitKey(0)  # Wait for a key press to close
        cv2.destroyAllWindows()  # Close the window

def process_image(img):
    # Convert grayscale to RGB if necessary
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Convert to PyTorch tensor, rearrange channels, and normalize
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img

class ApplyKMeans():
    def __init__(self, dataset, model, frame_number):
        self.frame_number = str(frame_number)
        self.dataset = dataset
        self.player_images = self.dataset.frame_images[self.frame_number]
        self.embeddings = {}
        self.dataset = dataset
        self.model = model
        self.model.eval()

        embedding_list = []
        player_ids = []
        for player_id in self.player_images.keys():
            input = process_image(self.player_images[player_id][1])
            input = input.unsqueeze(0)
            embedding = self.model(input)
            embedding = embedding.detach().numpy()
            self.embeddings[player_id] = embedding
            embedding_list.append(embedding.squeeze())
            player_ids.append(player_id)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embedding_list)
        labels = kmeans.labels_

        # Group player IDs based on cluster assignment
        self.cluster_groups = {}
        for player_id, label in zip(player_ids, labels):
            self.cluster_groups[player_id] = label
    
    def display(self):
        display = DisplayFrame(self.dataset, self.frame_number)
        display.show_img(self.cluster_groups)


if __name__ == "__main__":
    dataset = TeamAssignmentDataset(is_train_set=False)
    frame_number = 35
    model = EmbeddingCNN()
    weights_path = "/Users/daniel/Desktop/Soccer-Team-Assignment/weights/model_weights.pth"
    model.load_state_dict(torch.load(weights_path))
    kmeans = ApplyKMeans(dataset, model, frame_number)
    kmeans.display()