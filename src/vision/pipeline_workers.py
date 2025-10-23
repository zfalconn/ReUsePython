import threading
from queue import Queue, Empty
import time
import numpy as np
import cv2
from asyncua.sync import Client, ThreadLoop

from src.vision.detection_fn import detection_xyz,draw_detection, colorize_depth
from src.vision.realsense_stream import RealSenseStream

# -------------------------
# Detection Worker
# -------------------------

class DetectionWorker(threading.Thread):
    def __init__(self, model, camera : RealSenseStream, max_queue_size=1, display=False, **yolo_args):
        super().__init__(daemon=True)
        self.model = model
        self.camera = camera  # 👈 full camera reference
        self.frame_queue = camera.frame_queue
        self.detections_queue = Queue(maxsize=max_queue_size)
        self.display = display
        self.running = True
        self.yolo_args = yolo_args

    def start(self):
        print("[DetectionWorker] started.")
        while self.running:
            try:
                color_frame, depth_frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Access camera constants directly 👇
            depth_scale = self.camera.depth_scale
            intrinsics = self.camera.depth_intrinsics
            width, height = self.camera.width, self.camera.height

            # Pass intrinsics to your detection function
            detections = detection_xyz(
                self.model,
                color_frame,
                depth_frame,
                intrinsics=intrinsics,
                img_width=width,
                img_height=height,
                **self.yolo_args
            )

            if not self.detections_queue.full():
                self.detections_queue.put(detections)

            if self.display:
                
                color_annotated = draw_detection(color_image, detections)
                cv2.imshow("YOLO Detections with XYZ coordinate", color_annotated)

                depth_colored = colorize_depth(depth_frame=depth_frame, depth_scale=depth_scale)
                cv2.imshow("Depth Map", depth_colored)

                if cv2.waitKey(1) == 27:  # ESC
                    self.running = False
                    break

        cv2.destroyAllWindows()
        print("[DetectionWorker] exiting.")

    def stop(self):
        self.running = False
        self.join()


# -------------------------
# Processing Worker
# -------------------------
class ProcessingWorker(threading.Thread):
    def __init__(self, det_queue, processed_queue, running_flag):
        super().__init__(daemon=True)
        self.det_queue = det_queue
        self.processed_queue = processed_queue
        self.running_flag = running_flag

    def process_detections(self, detections):
        # Example: Compute centroid or filter detections
        processed = []
        for det in detections:
            x, y, z = det["x"], det["y"], det["z"]
            processed.append((x, y, z))
        return processed

    def run(self):
        print("[Processing] Thread started.")
        while self.running_flag["run"]:
            try:
                detections = self.det_queue.get(timeout=1)
            except Empty:
                continue

            processed_coords = self.process_detections(detections)
            self.processed_queue.put(processed_coords)

        print("[Processing] Thread exiting.")

