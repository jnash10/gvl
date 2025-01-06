from anthropic import Client
from PIL import Image
from utils.image_utils import image_to_base64


class ClaudeVLM:
    def __init__(self, api_key):
        self.client = Client(api_key=api_key)
        self.model_name = "claude-3-5-sonnet-20241022"

    def get_single_prediction(
        self, frame: Image.Image, task_description: str, current_idx: int
    ) -> float:
        image_data = image_to_base64(frame)
        media_type = "image/jpeg"
        content = [
            {
                "type": "text",
                "text": f"Task Description: {task_description}\nEvaluate Frame {current_idx}:",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            },
        ]
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=100,
            messages=[{"role": "user", "content": content}],
        )
        response_text = message.content[0].parts[0].text
        percentage_str = response_text.strip().split()[-1]
        if percentage_str.endswith("%"):
            percentage = float(percentage_str.strip("%"))
        else:
            percentage = 50.0
        return percentage
