from ultralytics import YOLO

model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    model.train(data=r"data\Focus1.v1i.yolov11\data.yaml", 
                epochs=500, patience=100, batch=16, 
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"..\models")