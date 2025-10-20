from ultralytics import YOLO

image_path= r"..\test_imgs\7.jpg"
model = YOLO(r"..\models\TicTacToe_250722_s.pt")


results = model(source=image_path,conf=0.1,save=True)