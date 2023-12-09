import itertools
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random

class TeamAssignmentDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        """
        image_dir: Directory where images are stored.
        annotations: A list of tuples. Each tuple contains the image filename, bounding box, and team label.
        transform: PyTorch transforms for preprocessing.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.triplets = self.preprocess_annotations(annotations)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        anchor_info, positive_info, negative_info = self.triplets[index]
        anchor_image = self.load_image(*anchor_info)
        positive_image = self.load_image(*positive_info)
        negative_image = self.load_image(*negative_info)

        return anchor_image, positive_image, negative_image

    def load_image(self, filename, bbox):
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = image.crop(bbox)
        if self.transform:
            image = self.transform(image)
        return image

    def preprocess_annotations(self, annotations):
        """
        Process the annotations to generate all possible triplets for each image.

        annotations: A list of tuples. Each tuple contains the image filename, bounding box, and team label.

        Returns a list of triplets, where each triplet is a tuple of (anchor_info, positive_info, negative_info),
        and each *_info is a tuple of (filename, bbox).

        Explanation:
        - Grouping by Image: First, we group all annotations by their respective images and then further group them by teams within each image. This results in a dictionary where each key is an image filename, and the value is another dictionary with keys team_0 and team_1, containing the players' information.
        - Generating Triplets: For each image, we iterate through the players of each team. For each player (considered as the anchor), we pair them with every other player in the same team (as positive samples) and with each player in the opposing team (as negative samples).
        - Excluding Self Pairing: We ensure that the anchor and positive samples are not the same player.
        """
        triplets = []

        # Group annotations by image
        image_annotations = {}
        for filename, bbox, label in annotations:
            if filename not in image_annotations:
                image_annotations[filename] = {'team_0': [], 'team_1': []}
            team_key = 'team_0' if label == 0 else 'team_1'
            image_annotations[filename][team_key].append((filename, bbox))

        # Iterate over each image
        for filename in image_annotations:
            team_0_players = image_annotations[filename]['team_0']
            team_1_players = image_annotations[filename]['team_1']

            # Generate all possible combinations within each team
            for team_players, opposite_team_players in [(team_0_players, team_1_players), (team_1_players, team_0_players)]:
                for anchor in team_players:
                    for positive in team_players:
                        if anchor != positive:
                            # For each anchor-positive pair, pair with each player in the opposite team
                            for negative in opposite_team_players:
                                triplet = (anchor, positive, negative)
                                triplets.append(triplet)

        return triplets
