import threading
from queue import Queue, Empty, Full
import logging
import time
import numpy as np
import cv2
from asyncua.sync import Client, ThreadLoop

from src.vision.detection_fn import detection_xyz, detection_xyz_obb, draw_detection, draw_detection_obb, colorize_depth
from src.vision.realsense_stream import RealSenseStream
from ..utils.queue_helper import put_latest

# -------------------------
# Detection Worker
# -------------------------

class DetectionWorker(threading.Thread):
    def __init__(self, model, camera: RealSenseStream, max_queue_size=1, display=False, obb = False, limit_box=True, **yolo_args):
        super().__init__(daemon=True)
        self.model = model
        self.camera = camera
        
        self.frame_queue = camera.frame_queue
        self.detections_queue = Queue(maxsize=max_queue_size)
        self.annotated_image_queue = Queue(maxsize=max_queue_size)

        self._display = display
        self._limit_box = limit_box
        self._obb = obb

        self.running = False
        self.yolo_args = yolo_args

        self.det_logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        self.running = True
        depth_scale = self.camera.depth_scale
        intrinsics = self.camera.depth_intrinsics
        width, height = self.camera.width, self.camera.height
        self.det_logger.info("Thread started")
        while self.running:
            try:
                frame = self.camera.get_latest_frame()
                if frame is None:
                    continue
                color_frame, depth_frame = frame

            except Empty:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            if not self._obb:
                detections = detection_xyz(
                    self.model,
                    color_frame,
                    depth_frame,
                    intrinsics=intrinsics,
                    img_width=width,
                    img_height=height,
                    **self.yolo_args
                )

                if self._display:
                    color_annotated = draw_detection(color_image, detections, self._limit_box)

                    depth_colored = colorize_depth(depth_frame=depth_frame, depth_scale=depth_scale)
                else:
                    color_annotated = color_image
                    depth_colored = depth_frame
            
            if self._obb:
                detections = detection_xyz_obb(
                    self.model,
                    color_frame,
                    depth_frame,
                    intrinsics=intrinsics,
                    img_width=width,
                    img_height=height,
                    **self.yolo_args
                )

                if self._display:
                    color_annotated = draw_detection_obb(color_image, detections, self._limit_box)

                    depth_colored = colorize_depth(depth_frame=depth_frame, depth_scale=depth_scale)
                else:
                    color_annotated = color_image
                    depth_colored = depth_frame

            put_latest(self.detections_queue, detections)
            put_latest(self.annotated_image_queue, (color_annotated,depth_colored))

        self.det_logger.info("Detection stop")

    def stop(self):
        self.running = False
        self.join()

class DisplayWorker(threading.Thread):
    def __init__(self, annotated_image_queue : Queue):
        super().__init__(daemon=True)
        self.running = False
        self._annotated_image_queue = annotated_image_queue

        self.display_logger = logging.getLogger(self.__class__.__name__)

        
    def run(self):
        self.running = True
        self.display_logger.info("Thread start")
        while self.running:
            try:
                annotated_image = self._annotated_image_queue.get()
                annotated_color, annotated_depth = annotated_image
            except Empty:
                continue
            cv2.imshow("YOLO Detections with XYZ coordinate", annotated_color)
            cv2.imshow("Depth Map", annotated_depth)
            
            if cv2.waitKey(1) == 27:  # ESC
                self.running = False
                break
            
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.join()
        self.display_logger.info("Thread stop")