from ultralytics import YOLO

model = YOLO(r"models\focus1\retrain_obb_251113\train6\weights\morrow_obb_251119.pt")

if __name__ == '__main__':
    model.train(data=r"data\annotated\Focus1-BIGMAP.v2-morrow_251020.yolov11\data.yaml", 
                epochs=500, patience=50, batch=32, imgsz=640, cache = 'disk',
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"models\focus1\retrain_obb_BIGMAP_251203")

