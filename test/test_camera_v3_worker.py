import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO

import sys
import os
# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import DetectionWorker
from src.communication.opcua_device import PLCClient, Yaskawa_YRC1000


def main():
    # -----------------------------
    # Initialize shared state
    # -----------------------------
    running_flag = {"run": True}

    # -----------------------------
    # Initialize camera
    # -----------------------------
    camera = RealSenseStream(fps=30, width=1280, height=720)
    camera.start()
    # -----------------------------
    # Load your YOLO or detection model
    # -----------------------------
    model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")

    #model = YOLO(r"models\focus1\topdetectbest.pt")
   
    # -----------------------------
    # Initialize OPCUA Clients
    # -----------------------------

    plc_url = "opc.tcp://192.168.0.1:4840"
    robot_url = "opc.tcp://192.168.0.20:16448"
    #robot = Yaskawa_YRC1000(robot_url)
    plc = PLCClient(plc_url)
    
    # -----------------------------
    # Initialize workers
    # -----------------------------

    detection_worker = DetectionWorker(
        model=model,
        camera=camera, running_flag=running_flag,
        max_queue_size=2,
        display=True,   # set False for headless
        conf=0.8  # example YOLO argument
    )

    # -----------------------------
    # Start workers
    # -----------------------------

    
    detection_worker.start()
   # print(robot.get_available_jobs())
    #print(robot.set_servo(True))
    print("[Main] All threads started. Press Ctrl+C to stop.")

    # -----------------------------
    # Main loop (could be your PLC sender, logging, etc.)
    # -----------------------------
    try:
        while running_flag["run"]:
            try:
                detections = detection_worker.detections_queue.get(timeout=0.2)
            except Empty:
                continue

            housing_detection = next(
                (det for det in detections if det["class_name"] == "battery_housing"), 
                None
            )

            if housing_detection:
                if (640-300) < housing_detection["center_2d"][0] < (640+300) and (360-200) < housing_detection["center_2d"][1] < (360+200):

                    housing_coord_gripper = housing_detection["xyz_gripper_frame"]
                    print(f"Housing coord in gripper's frame: {housing_coord_gripper}")

                    # Send data to PLC
                    plc.send_coordinates(
                        x=housing_coord_gripper[0]*1000,
                        y=0,
                        z=housing_coord_gripper[2]*1000
                    )

                    time.sleep(0.1)
                    plc.set_trigger(True)
                    time.sleep(0.1)
                    plc.set_trigger(False)
                

    except KeyboardInterrupt:
        print("[Main] Keyboard interrupt detected. Stopping...")

    finally:
        print("[Main] Cleaning up...")
        plc.send_coordinates(
                    x=0,
                    y=0,
                    z=0
                )
        running_flag["run"] = False
        detection_worker.stop()
        camera.stop()
        plc.stop_communication()
        print("[Main] All threads stopped cleanly.")


if __name__ == "__main__":
    main()
