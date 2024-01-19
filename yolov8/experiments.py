from ultralytics import YOLO
from time import time


if __name__ == "__main__":

    start = time()
    model = YOLO("yolov8n.pt")
    results = model.train(data="screw_data.yaml",
                          epochs=50, batch=16,imgsz=1024 , workers=8, project="n_50_16_1024")
    results = model.train(data="screw_data.yaml",
                          epochs=100, batch=16,imgsz=1024 , workers=8, project="n_100_16_1024")
    results = model.train(data="screw_data.yaml",
                          epochs=200, batch=16, imgsz=1024 ,workers=8, project="n_200_16_1024")

    results = model.train(data="screw_data.yaml",
                          epochs=50, batch=32, imgsz=1024 ,workers=8, project="n_50_32_1024")
    results = model.train(data="screw_data.yaml",
                          epochs=100, batch=32,imgsz=1024 , workers=8, project="n_100_32_1024")
    results = model.train(data="screw_data.yaml",
                          epochs=200, batch=32,imgsz=1024 , workers=8, project="n_200_32_1024")
    
    end = time() - start
    print(end)
    