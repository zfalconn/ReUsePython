import time
import threading
from queue import Queue
from ultralytics import YOLO

import sys
import os
# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import DetectionWorker, ProcessingWorker


def main():
    # -----------------------------
    # Initialize shared state
    # -----------------------------
    running_flag = {"run": True}

    # Queues connecting each stage
    detection_queue = Queue(maxsize=2)
    #processed_queue = Queue(maxsize=2)

    # -----------------------------
    # Initialize camera
    # -----------------------------
    camera = RealSenseStream(fps=30, width=1280, height=720)
    camera.start()

    # -----------------------------
    # Load your YOLO or detection model
    # -----------------------------
    model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")

    # -----------------------------
    # Start workers
    # -----------------------------
    detection_worker = DetectionWorker(
        model=model,
        camera=camera,
        max_queue_size=2,
        display=True,   # set False for headless
        conf=0.8  # example YOLO argument
    )

    # processing_worker = ProcessingWorker(
    #     det_queue=detection_worker.detections_queue,
    #     processed_queue=processed_queue,
    #     running_flag=running_flag
    # )

    detection_worker.start()
    # processing_worker.start()

    print("[Main] All threads started. Press Ctrl+C to stop.")

    # -----------------------------
    # Main loop (could be your PLC sender, logging, etc.)
    # -----------------------------
    try:
        while running_flag["run"]:
            try:
                #processed_data = processed_queue.get(timeout=1)
                #print("[Main] Processed data:", processed_data)
                # → here you can forward to OPC UA, PLC, etc.
                print("Do some process")
            except Exception:
                pass
    except KeyboardInterrupt:
        print("[Main] Stopping...")
        running_flag["run"] = False
        detection_worker.stop()
        camera.stop()

    print("[Main] All threads stopped cleanly.")

if __name__ == "__main__":
    main()
