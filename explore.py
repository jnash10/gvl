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


class FrameReader:
    """Handles reading frames from directory."""

    @staticmethod
    def read_frames_from_dir(
        dir_path: str, ext: tuple = (".jpg", ".png", ".jpeg")
    ) -> List[Image.Image]:
        """Read frames from directory in sorted order.

        Args:
            dir_path: Path to directory containing frames
            ext: Tuple of valid image extensions

        Returns:
            List of PIL Image objects
        """
        frame_files = [f for f in os.listdir(dir_path) if f.lower().endswith(ext)]
        frame_files.sort()  # Ensure frames are in order

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(dir_path, frame_file)
            frame = Image.open(frame_path)
            frames.append(frame)

        return frames


class VLMBase(ABC):
    """Base class for VLM implementations with auto-regressive prediction."""

    def __init__(self):
        self.base_prompt = """You are an expert at estimating task completion percentage from robot task execution frames.
        Given a sequence of frames from a robot task execution, estimate the completion percentage (0-100%) 
        for each frame. The sequence has been shuffled, so you need to carefully analyze each frame's state 
        to determine how far along the task has progressed.

        Task: {task_description}

        Previously analyzed frames and their completion percentages:
        {previous_predictions}

        For the current frame, provide your completion percentage estimate (0-100).
        Consider both the current frame's state and the previous predictions to ensure consistency.
        Respond ONLY with the percentage number, no other text.
        """

    def format_previous_predictions(
        self, frames_so_far: List[Image.Image], predictions_so_far: List[float]
    ) -> str:
        """Format previous predictions for prompt."""
        if not predictions_so_far:
            return "No previous frames analyzed yet."

        # Format each previous prediction
        formatted = []
        for i, pred in enumerate(predictions_so_far):
            formatted.append(f"Frame {i+1}: {pred:.1f}%")
        return "\n".join(formatted)

    def get_predictions(
        self, frames: List[Image.Image], task_description: str
    ) -> List[float]:
        """Get completion percentage predictions for frames in auto-regressive manner."""
        predictions = []
        frames_so_far = []

        for i, frame in enumerate(frames):
            # Format prompt with previous predictions
            prompt = self.base_prompt.format(
                task_description=task_description,
                previous_predictions=self.format_previous_predictions(
                    frames_so_far, predictions
                ),
            )

            # Get single prediction
            pred = self.get_single_prediction(frame, prompt)
            predictions.append(pred)

            # Update context for next prediction
            frames_so_far.append(frame)

        return predictions

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io
        import base64

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    @abstractmethod
    def get_single_prediction(self, frame: Image.Image, prompt: str) -> float:
        """Get prediction for a single frame. To be implemented by child classes."""
        pass


class ClaudeVLM(VLMBase):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = anthropic.Anthropic()

    def get_single_prediction(self, frame: Image.Image, prompt: str) -> float:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._image_to_base64(frame),
                        },
                    },
                ],
            }
        ]

        response = self.client.messages.create(
            model="claude-3-opus-20240229", max_tokens=100, messages=messages
        )

        try:
            prediction = float(response.content[0].text.strip())
            return prediction
        except ValueError:
            print(f"Warning: Could not parse prediction: {response.content[0].text}")
            return 50.0  # Default value


class OpenAIVLM(VLMBase):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)

    def get_single_prediction(self, frame: Image.Image, prompt: str) -> float:
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_base64_url(frame),
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )

        try:
            prediction = float(response.choices[0].message.content.strip())
            return prediction
        except ValueError:
            print(
                f"Warning: Could not parse prediction: {response.choices[0].message.content}"
            )
            return 50.0  # Default value


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

    def shuffle_frames(
        self, frames: List, keep_first: bool = True
    ) -> Tuple[List, List]:
        """Shuffle frames while optionally keeping first frame fixed."""
        if keep_first:
            first_frame = frames[0]
            remaining_frames = frames[1:]
            shuffled_indices = list(range(1, len(frames)))
            random.shuffle(shuffled_indices)

            shuffled_frames = [first_frame] + [frames[i] for i in shuffled_indices]
            indices = [0] + shuffled_indices
        else:
            shuffled_indices = list(range(len(frames)))
            random.shuffle(shuffled_indices)
            shuffled_frames = [frames[i] for i in shuffled_indices]
            indices = shuffled_indices

        return shuffled_frames, indices

    def predict(
        self, frames: List, task_description: str, keep_first: bool = True
    ) -> Tuple[List[float], List[int]]:
        """Main prediction pipeline."""
        # Subsample frames
        frames = self.subsample_frames(frames)

        # Shuffle frames
        shuffled_frames, indices = self.shuffle_frames(frames, keep_first)

        # Get predictions from VLM
        predictions = self.vlm.get_predictions(shuffled_frames, task_description)

        return predictions, indices


class GVLVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot_predictions(
        self,
        frames: List,
        predictions: List[float],
        indices: List[int],
        save_path: Optional[str] = None,
    ):
        """Plot frames and predictions."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # Plot task progress curve
        original_order_preds = [
            predictions[indices.index(i)] for i in range(len(frames))
        ]
        ax1.plot(
            range(len(frames)), original_order_preds, "b-", label="Predicted Progress"
        )
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Task Completion %")
        ax1.set_title("Predicted Task Progress")
        ax1.grid(True)
        ax1.legend()

        # Plot shuffle mapping
        ax2.scatter(range(len(indices)), indices, alpha=0.5)
        ax2.set_xlabel("Input Frame Position")
        ax2.set_ylabel("Shuffled Position")
        ax2.set_title("Frame Shuffle Mapping")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


def main():
    # Example usage
    FRAMES_DIR = "data/frames/fold_dress"
    TASK_DESCRIPTION = "The robot is folding a black dress"
    CLAUDE_API_KEY = os.environ["ANTHROPIC_API_KEY"]

    # Read frames
    frames = FrameReader.read_frames_from_dir(FRAMES_DIR)

    # Initialize VLM (choose one)
    vlm = ClaudeVLM(CLAUDE_API_KEY)
    # vlm = OpenAIVLM("your-openai-api-key")

    # Initialize predictor and visualizer
    predictor = GVLPredictor(vlm)
    visualizer = GVLVisualizer()

    # Get predictions
    predictions, indices = predictor.predict(frames, TASK_DESCRIPTION)

    # Visualize
    visualizer.plot_predictions(frames, predictions, indices)


if __name__ == "__main__":
    main()
