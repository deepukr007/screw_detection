from ultralytics import YOLO


if __name__ == "__main__":

    epochs = 100
    patience = epochs
    dataset_screw = '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/screw_data.yaml'
    dataset_weee = '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/WEEE_data.yaml'
    model = YOLO("yolov8n.pt")


    results = model.tune(data=dataset_weee,
                          epochs=epochs, iterations=100, optimizer='AdamW', batch=16, imgsz=1280
                            ,  workers=8, patience=patience, device='cuda:0' ,project="hp_weee" )
  
    