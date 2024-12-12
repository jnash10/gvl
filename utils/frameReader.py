from typing import List, Tuple, Optional, Dict
from PIL import Image
import os


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
