from ultralytics import YOLO
import os


if __name__ == "__main__":
    model_path = os.path.join("models", "best_epoch_50.pt")
    model = YOLO(model_path)
    results = model.val(split="test", workers=1)
