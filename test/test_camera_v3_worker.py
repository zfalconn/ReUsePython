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
    running_flag = {"run": True}
    camera = RealSenseStream(fps=30, width=1280, height=720)
    camera.start()

    model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")
    detection_worker = DetectionWorker(
        model=model,
        camera=camera,
        running_flag=running_flag,
        max_queue_size=2,
        display=True,
        conf=0.8
    )
    detection_worker.start()

    plc_url = "opc.tcp://192.168.0.1:4840"
    robot_url = "opc.tcp://192.168.0.20:16448"

    # Use context managers for OPCUA clients
    with Yaskawa_YRC1000(robot_url) as robot, PLCClient(plc_url) as plc:
        print(robot.get_available_jobs())
        print("[Main] All threads started. Press Ctrl+C to stop.")

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

                #print(robot.start_job('CAM_HOME', block=True))
                #time.sleep(2)

                if housing_detection:
                    cx, cy = housing_detection["center_2d"]
                    if (640-300) < cx < (640+300) and (360-200) < cy < (360+200):
                        housing_coord_gripper = housing_detection["xyz_gripper_frame"]
                        print(f"Housing coord in gripper's frame: {housing_coord_gripper}")

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
            plc.send_coordinates(x=0, y=0, z=0)
            running_flag["run"] = False
            detection_worker.stop()
            camera.stop()
            print("[Main] All threads stopped cleanly.")


if __name__ == "__main__":
    main()
