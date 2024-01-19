from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    results = model.train(data="screw_data.yaml",
                          epochs=5, batch=4, workers=1, project="m_5_4")
