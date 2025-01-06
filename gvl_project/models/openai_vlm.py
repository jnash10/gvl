import openai
from PIL import Image
from utils.image_utils import image_to_base64


class OpenAIVLM:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.model_name = "gpt-4o-mini"

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
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}",
                },
            },
        ]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
        )
        response_text = response.choices[0].message.content
        percentage_str = response_text.strip().split()[-1]
        if percentage_str.endswith("%"):
            percentage = float(percentage_str.strip("%"))
        else:
            percentage = 50.0
        return percentage
