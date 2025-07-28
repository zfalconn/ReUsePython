
import sys
import os
import cv2
import threading
from ultralytics import YOLO
from queue import Queue

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.detection_fn import detection_xyz, draw_detections
from src.vision.object_detection import ObjectDetection

def main():
    try:
        camera = RealSenseStream(fps=30, max_queue_size=5)
        camera.start()
        
        model = YOLO(r"..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")
        object_detector = ObjectDetection(model, display=True, max_queue_size=1, conf=0.7, device=0, max_det=5)
        
        object_detector.start(camera.get_frame_queue())

        print("Press ESC in display window to exit")

        # Wait here until detection/display loop stops running
        while object_detector.running:
            # Sleep a bit to avoid busy waiting
            import time
            time.sleep(0.1)

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        print("Stopping threads...")
        object_detector.stop()
        camera.stop()
        print("Exited cleanly")

if __name__ == "__main__":
    main()