from ultralytics import YOLO
import config
import json

if __name__ == "__main__":
    model = YOLO(config.MODEL_PATH)

    metrics = model.val(data=config.DATA_PATH, imgsz = 640, batch=1, device=0)

    # Convert metrics to dict and save
    metrics_dict = metrics.results_dict
    with open("metrics_focus1_n.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print("Metrics saved to metrics.json")