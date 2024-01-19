from pprint import pprint
from ultralytics import YOLO
import os
from time import time 
from glob import glob

if __name__ == "__main__":
    model_path = "/home/krishnar@iff.intern/thesis/screw_detection/yolov8/paper/train3/weights"
    for model in glob((os.path.join(model_path , "**/*.pt")), recursive=True):
            print(model)
            model = YOLO(model)
            start = time()
            results = model.val(split="test" , workers=8 ,save_json=True , device="cuda:0")
            end = time() - start 
            fps = 90 / end 
            print(fps)
