import numpy as np
import random


class DataProcessor:
    def __init__(self, num_frames=30):
        self.num_frames = num_frames

    def subsample_frames(self, frames: list) -> list:
        if len(frames) <= self.num_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        return [frames[i] for i in indices]

    def shuffle_frames(self, frames: list) -> tuple:
        initial_frame = frames[0]
        remaining_frames = frames[1:]
        shuffled_indices = list(range(1, len(frames)))
        random.shuffle(shuffled_indices)
        shuffled_indices = [0] + shuffled_indices
        shuffled_frames = [frames[i] for i in shuffled_indices]
        return shuffled_frames, shuffled_indices

    def map_predictions(
        self, shuffled_predictions: list, shuffle_indices: list, num_frames: int
    ) -> list:
        unshuffled_predictions = [None] * num_frames
        unshuffled_predictions[0] = 0.0
        for i, orig_idx in enumerate(shuffle_indices):
            unshuffled_predictions[orig_idx] = shuffled_predictions[i]
        return unshuffled_predictions
