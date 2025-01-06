from utils.frameReader import FrameReader
from predictors.gvl_predictor import GVLPredictor
from PIL import Image

# Load your frames (list of PIL Images)

frames = FrameReader.read_frames_from_dir(
    "/Users/agam/projects/gvl/data/frames/fold_dress"
)

task_description = "Task of assembling a robot"

predictor = GVLPredictor(vlm="claude", num_frames=10)
predictions = predictor.predict(frames, task_description)

print(predictions)
