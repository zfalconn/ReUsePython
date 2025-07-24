
import sys
import os
import cv2
import threading
from ultralytics import YOLO

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.RealSenseStream import RealSenseStream
from src.vision.detection import detection_xyz, draw_detections
import cv2

def display_loop(camera, model, running_flag):
    while running_flag["run"]:
        frames = camera.get_latest_frame() #Grab frame from Queue produced by camera.start()
        if frames is not None:
            color_frame, depth_frame = frames
            detections = detection_xyz(model, color_frame, depth_frame, confidence=0.5)
            color_frame = draw_detections(color_frame, detections)
            cv2.imshow("Color", color_frame)

        if cv2.waitKey(1) == 27:
            running_flag["run"] = False
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    # Test decoupling of camera acquisitions and detection
    # Further development can put camera and detection on different processes
    # Refactor code for cleaner variables calling such as model definition, 
    # which can be moved to a separate file or some kind of config
    # 

    ### INIT CAMERA AND DETECTION MODEL ###
    camera = RealSenseStream(fps=30)
    model = YOLO(r"..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")
    ### ----- ----- ----- ###

    ### START ACQUISTION THREAD ###
    camera.start() 
    ### ----- ----- ----- ###

    ### START DETECTION THREAD ###
    running_flag = {"run": True}
    display_thread = threading.Thread(target=display_loop, args=(camera,model,running_flag), daemon=True)
    display_thread.start()
    ### ----- ----- ----- ###
    try:
        while display_thread.is_alive():
            display_thread.join(timeout=1)
    except KeyboardInterrupt:
        print("Exiting...")

    camera.stop()
    cv2.destroyAllWindows()