import anthropic
import openai
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from PIL import Image
import time
import matplotlib.pyplot as plt


class VLMBase(ABC):
    """Base class for VLM implementations with auto-regressive prediction."""

    def __init__(self):
        self.base_prompt = """You are an expert roboticist tasked to predict task completion percentages for frames of a robot for the task of {task_description}.
The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. Note that these frames are in random order, so please pay attention to the individual frames when reasoning about task completion percentage.

Initial robot scene: [Initial Frame]
In the initial robot scene, the task completion percentage is 0.

Now, for the task of {task_description}, I will show you all frames in a randomly shuffled order. Previous predictions for some frames are provided.
Your task is to predict the completion percentage for Frame {current_frame_idx}.

{all_frames_prompt}

Previous predictions:
{previous_predictions}

Provide the completion percentage for Frame {current_frame_idx} following this format:
Frame {current_frame_idx}: Frame Description: [Description], Task Completion Percentages: X%"""

    def format_all_frames_prompt(
        self, all_frames: List[Image.Image], current_idx: int
    ) -> str:
        """Format the prompt section showing all frames."""
        return "\n".join(
            [f"Frame {i+1}: [Frame {i+1}]" for i in range(len(all_frames))]
        )

    def format_previous_predictions(self, predictions_so_far: List[float]) -> str:
        """Format previous predictions for prompt."""
        if not predictions_so_far:
            return "No previous predictions available."

        formatted = []
        for i, pred in enumerate(predictions_so_far):
            if pred is not None:  # Only include frames that have been predicted
                formatted.append(
                    f"Frame {i+1}: Task Completion Percentages: {pred:.1f}%"
                )
        return "\n".join(formatted)

    def get_predictions(
        self,
        shuffled_frames: List[Image.Image],
        task_description: str,
        initial_frame: Image.Image,
    ) -> List[float]:
        """Get completion percentage predictions for frames in auto-regressive manner.

        Args:
            shuffled_frames: List of frames in shuffled order
            task_description: Description of the task
            initial_frame: The first frame (unshuffled) to use as anchor

        Returns:
            List of predictions in the same order as shuffled_frames
        """
        predictions = [None] * len(shuffled_frames)  # Initialize with None

        # Make predictions one by one
        for i in range(len(shuffled_frames)):
            # Format prompt with all frames and previous predictions
            prompt = self.base_prompt.format(
                task_description=task_description,
                current_frame_idx=i + 1,
                all_frames_prompt=self.format_all_frames_prompt(shuffled_frames, i),
                previous_predictions=self.format_previous_predictions(predictions),
            )

            # Get prediction for current frame
            pred = self.get_single_prediction(
                current_frame=shuffled_frames[i],
                all_frames=shuffled_frames,
                initial_frame=initial_frame,
                prompt=prompt,
                current_idx=i,
            )
            predictions[i] = pred
            print(pred)
            plt.imshow(shuffled_frames[i])

            # Sleep for a bit to avoid rate limiting
            time.sleep(10)

        return predictions

    @abstractmethod
    def get_single_prediction(
        self,
        current_frame: Image.Image,
        all_frames: List[Image.Image],
        initial_frame: Image.Image,
        prompt: str,
        current_idx: int,
    ) -> float:
        """Get prediction for a single frame. To be implemented by child classes."""
        pass


class ClaudeVLM(VLMBase):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = anthropic.Anthropic()

    def get_single_prediction(
        self,
        current_frame: Image.Image,
        all_frames: List[Image.Image],
        initial_frame: Image.Image,
        prompt: str,
        current_idx: int,
    ) -> float:
        # Build content list with all images and text
        content = []

        # Split prompt and add images at appropriate locations
        parts = prompt.split("[Initial Frame]")
        content.append({"type": "text", "text": parts[0]})

        # Add initial frame
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": self._image_to_base64(initial_frame),
                },
            }
        )

        # Add middle text
        remaining_text = parts[1]

        # Add all frames section
        for i, frame in enumerate(all_frames):
            frame_marker = f"[Frame {i+1}]"
            if frame_marker in remaining_text:
                text_before, remaining_text = remaining_text.split(frame_marker, 1)
                if text_before:
                    content.append({"type": "text", "text": text_before})
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._image_to_base64(frame),
                        },
                    }
                )

        # Add final text
        if remaining_text:
            content.append({"type": "text", "text": remaining_text})

        messages = [{"role": "user", "content": content}]

        response = self.client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=100, messages=messages
        )

        try:
            response_text = response.content[0].text.strip()
            print(response_text)
            percentage_str = (
                response_text.split("Task Completion Percentages:")[-1]
                .strip()
                .rstrip("%")
            )
            prediction = float(percentage_str)
            print("current index", current_idx, "prediction", prediction)
            return prediction
        except ValueError:
            print(f"Warning: Could not parse prediction: {response.content[0].text}")
            # extract number from response
            import re

            response_text = response.content[0].text.strip()
            percentage_str = re.findall(r"\d+", response_text)

            try:
                prediction = float(percentage_str[0])
                return prediction

            except:
                return 50.0

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io
        import base64

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()


class OpenAIVLM(VLMBase):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)

    def get_single_prediction(
        self,
        current_frame: Image.Image,
        all_frames: List[Image.Image],
        initial_frame: Image.Image,
        prompt: str,
        current_idx: int,
    ) -> float:
        # Build content list with all images and text
        content = []

        # Split prompt and add images at appropriate locations
        parts = prompt.split("[Initial Frame]")
        content.append({"type": "text", "text": parts[0]})
        # print(type(parts[0]), parts[0])

        # Add initial frame
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._image_to_base64(initial_frame)}",
                },
            }
        )

        # Add middle text
        remaining_text = parts[1]
        # print(remaining_text)

        # Add all frames section
        for i, frame in enumerate(all_frames):
            frame_marker = f"[Frame {i+1}]"
            if frame_marker in remaining_text:
                text_before, remaining_text = remaining_text.split(frame_marker, 1)
                if text_before:
                    content.append({"type": "text", "text": text_before})
                    # print(text_before)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._image_to_base64(frame)}",
                        },
                    }
                )
        # for _ in content:
        #     print(content)
        # Add final text
        if remaining_text:
            content.append({"type": "text", "text": remaining_text})
            # print(remaining_text)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                max_tokens=100,
            )

            response_text = response.choices[0].message.content.strip()
            percentage_str = (
                response_text.split("Task Completion Percentages:")[-1]
                .strip()
                .rstrip("%")
            )
            prediction = float(percentage_str)
            print("current index", current_idx, "prediction", prediction)
            return prediction
        except Exception as e:
            print(f"Warning: Error getting prediction: {str(e)}")
            return 50.0

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io
        import base64

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
