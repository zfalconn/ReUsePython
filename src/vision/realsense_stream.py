import threading
import logging
from queue import Queue, Full, Empty
from .realsense_frame import realsense_get_frame, realsense_init
from ..utils.queue_helper import put_latest

class RealSenseStream:
    """
    RealSenseStream class to initialize camera and start threaded acquistion
    """
    def __init__(self, width = 640, height = 480, fps = 15, enable_imu = False, max_queue_size=1):
        self.pipeline, self._depth_scale, self._depth_intrinsics = realsense_init(width, height, fps, enable_imu)
        self._frame_queue = Queue(maxsize=max_queue_size)
        self._width = width
        self._height = height
        self.running = False
        self.cam_logger = logging.getLogger(self.__class__.__name__)

    def _capture_loop(self):
        self.cam_logger.info("Capture loop started")
        while self.running:
            color_frame, depth_frame = realsense_get_frame(self.pipeline)

            if color_frame is not None and depth_frame is not None:
                try:
                    put_latest(self.frame_queue, (color_frame,depth_frame))
                except Exception as e:
                    logging.info(f'Exception from RealSenseStream capture loop: {e}')
                    continue

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.cam_logger.info("Thread started")

    def stop(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()
        self.cam_logger.info("Thread stopped")

    def get_latest_frame(self):
        latest = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except Empty:
                break
        return latest

    
    @property
    def depth_scale(self):
        return self._depth_scale

    @property
    def depth_intrinsics(self):
        return self._depth_intrinsics

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def frame_queue(self):
        return self._frame_queue
