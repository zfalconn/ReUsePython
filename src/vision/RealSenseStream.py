import threading
from queue import Queue
from .realsense_frame import realsense_get_frame, realsense_init

class RealSenseStream:
    """
    RealSenseStream class to start threaded acquistion
    """
    def __init__(self, width = 640, height = 480, fps = 15, enable_imu = False, max_queue_size=1):
        self.pipeline, self.depth_scale = realsense_init(width, height, fps, enable_imu)
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.running = False

    def _capture_loop(self):
        while self.running:
            color_frame, depth_frame = realsense_get_frame(self.pipeline)

            if color_frame is not None and depth_frame is not None:
                if not self.frame_queue.full():
                    # Add frame to queue
                    self.frame_queue.put((color_frame,depth_frame))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.pipeline.stop()

    def get_latest_frame(self):
        latest = None
        while not self.frame_queue.empty():
            latest = self.frame_queue.get()
        return latest

