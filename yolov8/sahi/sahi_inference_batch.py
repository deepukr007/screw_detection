from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path

# Download YOLOv8 model
model_path = "/home/krishnar@iff.intern/thesis/screw_detection/yolov8/models/trained/best.pt"
model_type = "yolov8"
model_device = "cuda:0" 
model_confidence_threshold = 0.4

slice_height = 1024
slice_width = 1024
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "/home/data/iDear/Own/"

result = predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)

