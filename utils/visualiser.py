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


class GVLVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot_predictions(
        self,
        frames: List,
        shuffled_predictions: List[float],
        unshuffled_predictions: List[float],
        shuffle_indices: List[int],
        save_path: Optional[str] = None,
    ):
        """Plot frames and predictions.

        Args:
            frames: Original frame sequence
            shuffled_predictions: Predictions in shuffled order
            unshuffled_predictions: Predictions in original frame order
            shuffle_indices: Indices used for shuffling
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize)

        # Plot unshuffled task progress curve
        ax1.plot(
            range(len(frames)),
            unshuffled_predictions,
            "b-",
            label="Predicted Progress (Original Order)",
        )
        ax1.set_xlabel("Original Frame Number")
        ax1.set_ylabel("Task Completion %")
        ax1.set_title("Predicted Task Progress (Original Order)")
        ax1.grid(True)
        ax1.legend()

        # Plot shuffled predictions
        ax2.plot(
            range(len(frames)),
            shuffled_predictions,
            "r-",
            label="Predicted Progress (Shuffled Order)",
        )
        ax2.set_xlabel("Shuffled Frame Number")
        ax2.set_ylabel("Task Completion %")
        ax2.set_title("Predicted Task Progress (Shuffled Order)")
        ax2.grid(True)
        ax2.legend()

        # Plot shuffle mapping
        ax3.scatter(range(len(shuffle_indices)), shuffle_indices, alpha=0.5)
        ax3.set_xlabel("Input Frame Position")
        ax3.set_ylabel("Shuffled Position")
        ax3.set_title("Frame Shuffle Mapping")
        ax3.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

        # Print Value-Order Correlation (VOC)
        voc = self.calculate_voc(unshuffled_predictions)
        print(f"\nValue-Order Correlation (VOC): {voc:.3f}")

    @staticmethod
    def calculate_voc(predictions: List[float]) -> float:
        """Calculate Value-Order Correlation (VOC) metric.

        Args:
            predictions: List of predictions in original frame order

        Returns:
            VOC score between -1 and 1
        """
        from scipy.stats import spearmanr

        # Create ordered indices (0 to len-1)
        ordered_indices = np.arange(len(predictions))

        # Calculate rank correlation
        correlation, _ = spearmanr(predictions, ordered_indices)
        return correlation
