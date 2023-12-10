from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import cv2
import numpy as np

"""
TODO:
- Implement train/test functionality with no frame leakage
- Try on test set !! (With k-means I assume? - Or at least the cosine similarity function)
- Add augmentations: create library of pairs in real time with augmentation (motion blur + grayscale)
- When making a multi-video dataset, switch the train_test split to be at the video and not at the frame...
"""


class TeamAssignmentDataset(Dataset):
    # Load in all frames corresponding to JSON from single video
    def __init__(self, is_train_set=True, dir="1"):
        """
        Expects JSON to contain bbox coords in x1, y1, x2, y2 format
        """

        self.data_dir = "/Users/daniel/Desktop/Soccer-Team-Assignment/data"
        self.dir = dir # our code only takes in a single repo at the moment

        self.is_train_set = is_train_set

        v_dir = os.path.join(self.data_dir, "videos/" + self.dir)
        l_dir = os.path.join(self.data_dir, "labels/" + self.dir)
        self.video_dir = self.get_file_in_folder(v_dir, "mp4")
        self.label_dir = self.get_file_in_folder(l_dir, "json")

        self.train_test_split = 0.8

        self.frame_images = {} # all images (cv2 format) stored here for each frame 
        self.triplets = []
        self.preprocess_annotations()
    
    def visualize_annotations(self):
        frame_images = []

        with open(self.label_dir) as json_file:
            data = json.load(json_file)

        last_frame = str(sorted([int(key) for key in data["frame_data"].keys()])[-1])

        frame_objects = data["frame_data"][last_frame]

        cap = cv2.VideoCapture(self.video_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(last_frame))
        ret, frame = cap.read() # read the last frame

        if not ret:
            print("Error reading the last frame from video")
            return

        # Process each object in the last frame
        print('Last frame is: ', last_frame)
        for obj_id, obj_data in frame_objects.items():
            print('Last frame object: ', obj_id)
            if obj_id in data['player_team_assignments'] and data['player_team_assignments'][obj_id] != -1:
                coords = obj_data["coords"]
                x1, y1, x2, y2 = coords
                cropped_img = frame[y1:y2, x1:x2]
                resized_img = cv2.resize(cropped_img, (120, 120))
                resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                frame_images.append((data['player_team_assignments'][obj_id], resized_img_rgb))
        print('Length of frame_images ', len(frame_images))

        cap.release()

        # START - Show single frame
        # img, value = frame_images[0]
        # plt.imshow(img)
        # plt.show()
        # END - Show single frame

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns for each team

        # Initialize lists to store images for each team
        team_images = {0: [], 1: [], 2: []}

        # Organize images by team
        for value, image in frame_images:
            team_images[value].append(image)

        # Display images for each team
        for i in range(3):
            if team_images[i]:  # Check if there are images for the team
                # Concatenate images horizontally and display
                concatenated_image = cv2.vconcat(team_images[i])
                axs[i].imshow(concatenated_image)
            else:
                # Display a blank image if no images for the team
                axs[i].imshow(np.zeros((120, 120, 3), dtype=np.uint8))

            axs[i].set_title(f"Team {i} Images")
            axs[i].axis('off')
        plt.subplots_adjust(hspace=0.2)
        plt.tight_layout()
        plt.show()

    def get_file_in_folder(self, folder_path, extension):
        """
        This function finds the file with the given extension in the folder
        params:
        @extension: either "txt" or "mp4"
        """
        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(f".{extension}"):
                file_path = os.path.join(folder_path, filename)
                return file_path
        # No file with the specified extension was found
        raise Exception()


    def __len__(self):
        return len(self.triplets)
    
    def preprocess_annotations(self):
        """
        - self.video_path contains the path for the video, please capture this with cv2
        - iterate through JSON and populate self.frame_images for each frame with images in cv2 format
        - populate self.triplets with pairings of images at each frame
        """
        # first, load in the JSON
        with open(self.label_dir) as json_file:
            data = json.load(json_file)
        annotations = data["player_team_assignments"]
        frame_data = data["frame_data"]

        train_test_idx = int(len(frame_data) * self.train_test_split)

        cap = cv2.VideoCapture(self.video_dir)

        # For each frame create dictionary holding all _relevant_ images (relevant if they are annotated)
        for i, frame in enumerate(frame_data):
            if self.is_train_set and i >= train_test_idx:
                continue
            if not self.is_train_set and i < train_test_idx:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ret, frame_img = cap.read() # read the current frame
            curr_frame_images = {} 
            for object_id, obj_data in frame_data[frame].items():
                if object_id in annotations and annotations[object_id] != -1:
                    coords = obj_data["coords"]
                    x1, y1, x2, y2 = coords
                    cropped_img = frame_img[y1:y2, x1:x2]
                    resized_img = cv2.resize(cropped_img, (120, 120))
                    resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                    curr_frame_images[object_id] = (annotations[object_id], resized_img_rgb)
            self.frame_images[frame] = curr_frame_images

        # Create dataset by calculating all combinations per frame (references to images through object_id + frame, not the actual image)

        annotated_players = set([object_id for object_id in annotations.keys() if annotations[object_id] != -1])

        # find index of frame to start test set
        frame_split_idx = int(len(frame_data) * self.train_test_split)

        for i, frame in enumerate(frame_data):
            if self.is_train_set and i >= train_test_idx:
                continue
            if not self.is_train_set and i < train_test_idx:
                continue
            # intersection of the annotated players and players in the frame
            players_in_frame = set(frame_data[frame].keys())
            intersection_set = annotated_players.intersection(players_in_frame)
            # create set for each of the 3 classes
            players = {0: [], 1: [], 2: []}
            for player in intersection_set:
                players[annotations[player]].append(player)
            new_triplets = self.create_triplet_combinations(players)
            new_triplets = [(frame, triplet) for triplet in new_triplets] # put the frame on each so we can add to the main triplets list
            self.triplets.extend(new_triplets)
        print('We finish with ', len(self.triplets), ' triplets')

    def create_triplet_combinations(self, players):
        """
        - We do not take an anchor for the Other class as we don't want a goalkeeper to have same embedding as a ref.
        Our objective in this sense is to only have the 
        """
        team1, team2, others = players[0], players[1], players[2] # each of these are sets with object_ids
        if len(team1) < 2 or len(team2) < 2 or len(others) < 2: # revise this statement
            return []
        triplets = []

        ### START TEST - minimal version
        # return [[team1[0], team1[1], team2[0]]] # format is anchor, positive, negative sample
        ### END TEST - minimal version

        # Generating triplets for team1 as anchor and team2 as negative
        for anchor in team1:
            for positive in set(team1) - {anchor}:
                for negative in team2:
                    triplets.append((anchor, positive, negative))

        # Generating triplets for team1 as anchor and others as negative
        for anchor in team1:
            for positive in set(team1) - {anchor}:
                for negative in others:
                    triplets.append((anchor, positive, negative))

        # Generating triplets for team2 as anchor and team1 as negative
        for anchor in team2:
            for positive in set(team2) - {anchor}:
                for negative in team1:
                    triplets.append((anchor, positive, negative))

        # Generating triplets for team2 as anchor and others as negative
        for anchor in team2:
            for positive in set(team2) - {anchor}:
                for negative in others:
                    triplets.append((anchor, positive, negative))
        return triplets

    def __getitem__(self, idx):
        """
        triplets list attribute has the reference to the image, so here we grab the image from the frame_images attribute!
        """
        frame, triplet_idx = self.triplets[idx]
        anchor_idx, positive_idx, negative_idx = triplet_idx # extract image indexes
        # retrieve images (in cv2 format) using indexes
        _, anchor = self.frame_images[frame][anchor_idx] # the _ is the class label e.g. team1, team2 , other
        _, positive = self.frame_images[frame][positive_idx]
        _, negative = self.frame_images[frame][negative_idx]

        return anchor, positive, negative
    
    def display_idx(self, idx):
        anchor, positive, negative = self.__getitem__(idx)
        images = [ anchor, positive, negative ]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns for each team
        for i in range(3):
            axs[i].imshow(images[i])
        axs[i].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dataset = TeamAssignmentDataset(is_train_set=True)
    # dataset.visualize_annotations()
    dataset.display_idx(0)