from utils.data_processor import DataProcessor
from models.claude_vlm import ClaudeVLM
from models.openai_vlm import OpenAIVLM


class GVLPredictor:
    def __init__(self, vlm: str, num_frames=30):
        self.data_processor = DataProcessor(num_frames=num_frames)
        if vlm == "claude":
            self.vlm_model = ClaudeVLM(api_key="your_api_key")
        elif vlm == "openai":
            self.vlm_model = OpenAIVLM(api_key="your_api_key")
        else:
            raise ValueError("Unsupported VLM model")

    def predict(self, frames: list, task_description: str) -> list:
        frames = self.data_processor.subsample_frames(frames)
        shuffled_frames, shuffle_indices = self.data_processor.shuffle_frames(frames)
        shuffled_predictions = []
        for idx, frame in enumerate(shuffled_frames, start=1):
            pred = self.vlm_model.get_single_prediction(frame, task_description, idx)
            shuffled_predictions.append(pred)
        unshuffled_predictions = self.data_processor.map_predictions(
            shuffled_predictions, shuffle_indices, len(frames)
        )
        return unshuffled_predictions
