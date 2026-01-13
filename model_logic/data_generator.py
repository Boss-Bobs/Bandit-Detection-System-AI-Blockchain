import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os

class VideoSequenceGenerator(Sequence):
    def __init__(self, video_paths_file, sequence_length, resize_dim=(128, 128), batch_size=1, shuffle=True): # <--- BATCH_SIZE=1, RESIZE_DIM=(128,128)
        self.video_paths = self._load_video_paths(video_paths_file)
        self.sequence_length = sequence_length
        self.resize_dim = resize_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end() # Call to shuffle data at the start

    def _load_video_paths(self, file_path):
        with open(file_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        return paths

    def _get_video_sequences(self, video_path):
        # This function loads frames from a single video and extracts all possible sequences
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.resize_dim)
            frame = frame / 255.0 # Normalize to [0, 1]
            frames.append(frame)
        cap.release()
        
        # Extract sequences (e.g., 8 frames each)
        sequences_from_video = []
        if len(frames) >= self.sequence_length:
            for i in range(0, len(frames) - self.sequence_length + 1):
                sequence = frames[i : i + self.sequence_length]
                sequences_from_video.append(np.array(sequence, dtype=np.float32))
        
        return sequences_from_video

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))


    def __getitem__(self, idx):
        batch_video_paths_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        all_sequences_in_batch = []
        for i in batch_video_paths_indices:
            video_path = self.video_paths[i]
            sequences_from_current_video = self._get_video_sequences(video_path)
            all_sequences_in_batch.extend(sequences_from_current_video)

        if not all_sequences_in_batch:
            dummy_input = np.zeros((self.batch_size, self.sequence_length, *self.resize_dim, 3), dtype=np.float32)
            return dummy_input, dummy_input

        np.random.shuffle(all_sequences_in_batch)
        final_batch_sequences = all_sequences_in_batch[:self.batch_size]
        
        if not final_batch_sequences:
            dummy_input = np.zeros((0, self.sequence_length, *self.resize_dim, 3), dtype=np.float32)
            return dummy_input, dummy_input

        batch_sequences_array = np.array(final_batch_sequences, dtype=np.float32)

        return batch_sequences_array, batch_sequences_array

    def on_epoch_end(self):
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
