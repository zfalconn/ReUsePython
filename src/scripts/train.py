from ultralytics import YOLO

model = YOLO("models/focus1/Focus1_YOLO11n_x1024_14112024.pt")

if __name__ == '__main__':
    model.train(data=r"data\annotated\Focus1-BIGMAP.v2-morrow_251020.yolov11\data.yaml", 
                epochs=500, patience=50, batch=16, imgsz=640, cache = True,
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"models\focus1\retrain")

