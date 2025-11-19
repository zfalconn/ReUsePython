from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")

if __name__ == '__main__':
    model.train(data=r"data\yolo_retrain_obb_v11_251113\Focus1.v2i.yolov11\data.yaml", 
                epochs=500, patience=50, batch=32, imgsz=640, cache = 'disk',
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"models\focus1\retrain_obb_251113")

