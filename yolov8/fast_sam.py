from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# Define an inference source
source = '/home/krishnar@iff.intern/thesis/datasets/crs/train/images/000014.jpg'

# Create a FastSAM model
model = FastSAM('/home/krishnar@iff.intern/thesis/screw_detection/yolov8/models/base/FastSAM-s.pt')  # or FastSAM-x.pt

everything_results = model(
    source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

ann = prompt_process.text_prompt(text='laptop')

prompt_process.plot(annotations=ann, output='./')
