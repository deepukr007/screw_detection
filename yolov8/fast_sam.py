from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# Define an inference source
source = '000001.jpg'

# Create a FastSAM model
model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

everything_results = model(
    source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

ann = prompt_process.text_prompt(text='a screw head')

prompt_process.plot(annotations=ann, output='./')
