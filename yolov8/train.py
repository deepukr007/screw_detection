from ultralytics import YOLO




if __name__ == "__main__":

    epochs = 500
    patience = epochs
    
    dataset_screw = '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/screw_data.yaml'
    dataset_weee = '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/WEEE_data.yaml'

    model = YOLO("yolov8n.pt")
    results = model.train(data=dataset_weee,
                          epochs=epochs, batch=32, imgsz=1280 ,  workers=8, patience=patience, device='cuda:0' ,optimizer="AdamW" 
                           , project="weee", name='500epochs_batch32' )
  
    print(results)
    print(results.results_dict['metrics/mAP50(B)'])