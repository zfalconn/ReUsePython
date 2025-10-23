from ultralytics import YOLO

image_path= r"..\test_imgs\7.jpg"
model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")
print(model.names)

#results = model(source=image_path,conf=0.1,save=True)