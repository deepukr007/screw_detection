from ultralytics import YOLO
import cv2 
from glob import glob
import os



if __name__ == "__main__":
    
    model = YOLO("/home/krishnar@iff.intern/thesis/screw_detection/yolov8/models/trained/best.pt")
    dataset_path = "/home/data/iDear/Own/"
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.mkdir(os.path.join(file_path , "labels" ))
    #results = model(source) 
    count = 0 
    file_tif  = glob(os.path.join(dataset_path ,'*.tif'))
    file_jpg = glob(os.path.join(dataset_path ,'*.jpg'))
    file = file_tif + file_jpg
    for image in file:
        image_name = str(os.path.split(image)[-1]).split('.')[0] 
        label_file = os.path.join(file_path , "labels" , f"{image_name}.txt")
        count = count + 1
        cv_image = cv2.imread(image)
        results = model.predict(cv_image, save=False, conf=0.1)
        boxes = results[0].boxes
        for index in range(len(boxes.cls)):
            x = boxes.xywhn[index][0].item()
            y = boxes.xywhn[index][1].item()
            w = boxes.xywhn[index][2].item()
            h = boxes.xywhn[index][3].item()
            string = f"{int(boxes.cls[index])} {x} {y} {w} {h} {boxes.conf[index]} \n"
            file = open(f"{label_file}" , 'a+') 
            file.write(string)
            file.close()
        
