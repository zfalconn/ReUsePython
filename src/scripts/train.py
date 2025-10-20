from ultralytics import YOLO

model = YOLO("models/focus1/Focus1_YOLO11s_x1024_14112024.pt")

if __name__ == '__main__':
    model.train(data=r"data\annotated\Focus1-BIGMAP.v2-morrow_251020.yolov11\data.yaml", 
                epochs=500, patience=100, batch=16, 
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"..\models\focus1")

