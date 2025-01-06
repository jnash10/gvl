import base64
from PIL import Image
import io


def image_to_base64(image: Image.Image, format="JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str
