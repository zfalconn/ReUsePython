import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO

import sys
import os
import logging
# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import DetectionWorker, DisplayWorker
from src.communication.opcua_device import PLCClient, Yaskawa_YRC1000


def main():
    
    camera = RealSenseStream(fps=30, width=1280, height=720)
    camera.start()

    model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")
    detection_worker = DetectionWorker(
        model=model,
        camera=camera,
        max_queue_size=1,
        display=True,
        conf=0.8
    )
    display_worker = DisplayWorker(detection_worker.annotated_image_queue)

    detection_worker.start()
    display_worker.start()

    try:
        while display_worker.running:
            time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("[Main] Keyboard interrupt detected. Stopping...")

    finally:
        logging.info("[Main] Cleaning up...")
        detection_worker.stop()
        camera.stop()
        logging.info("[Main] All threads stopped cleanly.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
