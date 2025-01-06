import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import random
import os
from PIL import Image
import json
from abc import ABC, abstractmethod
import anthropic
import openai
from pathlib import Path

from utils.vlm import VLMBase


class GVLPredictor:
    def __init__(self, vlm: VLMBase, num_frames: int = 30):
        """Initialize the GVL predictor.

        Args:
            vlm: VLM implementation to use
            num_frames: Number of frames to subsample to (default 30 as per paper)
        """
        self.vlm = vlm
        self.num_frames = num_frames

    def subsample_frames(self, frames: List) -> List:
        """Subsample frames to fixed length as mentioned in paper."""
        if len(frames) == self.num_frames:
            return frames

        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        return [frames[i] for i in indices]

    def shuffle_frames(self, frames: List) -> Tuple[List, List[int]]:
        """Shuffle frames while keeping first frame as anchor.

        Args:
            frames: List of frames to shuffle

        Returns:
            Tuple of (shuffled frames, shuffle indices)
        """
        # Keep first frame fixed
        first_frame = frames[0]
        remaining_frames = frames[1:]

        # Shuffle remaining frames
        shuffled_indices = list(range(1, len(frames)))
        random.shuffle(shuffled_indices)

        shuffled_indices = [0] + shuffled_indices

        # Create shuffled sequence
        shuffled_frames = [frames[i] for i in shuffled_indices]

        return shuffled_frames, shuffled_indices

    def predict(
        self, frames: List, task_description: str
    ) -> Tuple[List[float], List[int], List[float]]:
        """Main prediction pipeline.

        Args:
            frames: List of frames
            task_description: Description of the task

        Returns:
            Tuple of:
            - Predictions in shuffled order
            - Shuffle indices
            - Predictions in original order
        """
        # Subsample frames
        frames = self.subsample_frames(frames)

        # Get initial frame
        initial_frame = frames[0]

        # Shuffle remaining frames
        shuffled_frames, shuffle_indices = self.shuffle_frames(frames)

        # Get predictions from VLM (in shuffled order)
        shuffled_predictions = self.vlm.get_predictions(
            shuffled_frames=shuffled_frames,
            task_description=task_description,
            initial_frame=initial_frame,
        )

        # Reorder predictions to match original frame order
        unshuffled_predictions = [None] * len(frames)
        unshuffled_predictions[0] = 0.0  # Initial frame always 0%
        for i, orig_idx in enumerate(shuffle_indices):
            unshuffled_predictions[orig_idx] = shuffled_predictions[i]

        return shuffled_predictions, shuffle_indices, unshuffled_predictions
