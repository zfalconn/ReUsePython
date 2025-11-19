import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO
import numpy as np
import sys
import os
import logging
# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import DetectionWorker, DisplayWorker
from src.communication.opcua_device import PLCClient, Yaskawa_YRC1000
from src.vision.visual_controller import calc_control_val

def main():
    # --- Control parameters ---
    Kp = 0.2
    Kd = 0.4
    ALPHA = 0.32       # smoothing factor (0–1)
    DEADBAND_M = 0.02  # 10 mm deadband
    LOOP_HZ = 20
    LOOP_DT = 1.0 / LOOP_HZ

    # --- Connection URLs ---
    plc_url = "opc.tcp://192.168.0.1:4840"
    robot_url = "opc.tcp://192.168.0.20:16448"
    # postcp_index = 0
    # # --- Start camera and model ---
    # camera = RealSenseStream(fps=30, width=1280, height=720)
    # camera.start()

    # model = YOLO(r"models/focus1/retrain/train3/weights/best_morrow_251020.pt")

    # detection_worker = DetectionWorker(
    #     model=model,
    #     camera=camera,
    #     max_queue_size=1,
    #     display=True,
    #     limit_box=False,
    #     conf=0.8,
    #     imgsz=640
    # )
    # display_worker = DisplayWorker(detection_worker.annotated_image_queue)

    # detection_worker.start()
    # display_worker.start()

    # --- Initialize control state ---
    last_error = np.zeros(3)
    smoothed_error = np.zeros(3)

    with Yaskawa_YRC1000(robot_url) as robot, PLCClient(plc_url) as plc:
        try:
            print("robot initialized")
            print(robot.get_available_jobs())
            
            print(robot.set_servo(True))
            print(robot.start_job('BATTERY_HOME', block=True))
        #   print(robot.start_job("BASE2TOBASE1", block=True))
            print(robot.set_servo(False))
            robot.stop_communication()
            print('Program ended.')

        except KeyboardInterrupt:
            logging.info("[Main] Keyboard interrupt detected. Stopping...")

        finally:
            logging.info("[Main] Cleaning up...")
            logging.info("[Main] All threads stopped cleanly.")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
