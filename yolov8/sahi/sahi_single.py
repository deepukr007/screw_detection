from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
import os
from pprint import pprint

# Download YOLOv8 model
yolov8_model_path = "/home/krishnar@iff.intern/thesis/screw_detection/yolov8/models/trained/best.pt"
source_image_dir = "/home/krishnar@iff.intern/thesis/datasets/WDSD/test"
image = os.path.join(source_image_dir, "screws_185.png")



model_type = "yolov8"
model_path = yolov8_model_path
model_device = "cuda:0"  # or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 256
slice_width = 256
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2



detection_model = AutoDetectionModel.from_pretrained(
    model_type=model_type,
    model_path=yolov8_model_path,
    confidence_threshold=model_confidence_threshold,
    device=model_device,  # or 'cuda:0'
)

# Standard prediction
result = get_prediction(image, detection_model)
print(result.durations_in_seconds)

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1
)

# pprint(vars(result))

print(result.durations_in_seconds)
print(result.object_prediction_list)
