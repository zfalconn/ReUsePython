
import sys
import os
import cv2
import threading
from ultralytics import YOLO
import numpy as np

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.detection_fn import detection_xyz, draw_detections
import cv2

from src.vision.object_detection import ObjectDetection

def colorize_depth(depth_image, depth_scale, min_depth=0.2, max_depth=2.0):
    """
    Convert uint16 depth image (in camera units) to a colored 8-bit visualization.
    depth_scale: from pipeline.get_device().first_depth_sensor().get_depth_scale()
    """
    # Convert to meters
    depth_m = depth_image * depth_scale

    # Clip to range
    depth_clipped = np.clip(depth_m, min_depth, max_depth)

    # Normalize to 0–255
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colored


def display_loop(camera : RealSenseStream, model, running_flag):
    while running_flag["run"]:
        frames = camera.get_latest_frame()  # Grab frame from Queue produced by camera.start()
        if frames is not None:
            color_frame, depth_frame = frames

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Run detections
            detections = detection_xyz(model, color_image, depth_frame, conf=0.75)
            color_image = draw_detections(color_image, detections)

            # Display color frame
            cv2.imshow("Color", color_image)

            # --- Display depth frame ---

            depth_scale = camera.get_depth_scale() # store this when initializing your camera
            depth_colored = colorize_depth(depth_image, depth_scale)

            cv2.imshow("Depth", depth_colored)
            # ---------------------------

        if cv2.waitKey(1) == 27:  # ESC to stop
            running_flag["run"] = False
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    # Test decoupling of camera acquisitions and detection
    # Further development can put camera and detection on different processes
    # Refactor code for cleaner variables calling such as model definition, 
    # which can be moved to a separate file or some kind of config

    ### INIT CAMERA AND DETECTION MODEL ###
    camera = RealSenseStream(fps=30, width=1280, height=720)
    model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")
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