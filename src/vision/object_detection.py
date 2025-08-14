import threading
from queue import Queue, Empty
from .detection_fn import detection_xyz, draw_detections
from ultralytics import YOLO
import cv2
import time

class ObjectDetection:
    def __init__(self, model : YOLO, display : bool = False, max_queue_size = 1, **yolo_args):
        self.model = model
        self.frame_queue = None
        self.detections_queue = Queue(maxsize=max_queue_size)
        self.display = display
        self.running = False
        self.yolo_args = yolo_args
        
    
    
    # def _detection_loop(self):
    #     while self.running:
    #         try:
    #             color_frame, depth_frame = self.frame_queue.get(timeout=1)
    #         except Empty:
    #             continue
    #         if color_frame is None or depth_frame is None:
    #             continue

    #         detections = detection_xyz(self.model, color_frame, depth_frame, **self.yolo_args)
    #         if not self.detections_queue.full():
    #             self.detections_queue.put(detections)

    #         if self.display:
    #             annotated_frame = draw_detections(color_frame.copy(), detections)
    #             cv2.imshow("Bounding Box with XYZ", annotated_frame)
    #             if cv2.waitKey(1) == 27:  # ESC key to stop
    #                 self.running = False
    #                 break
    #     cv2.destroyAllWindows()
    
    def _detection_loop(self):
        while self.running:
            try:
                start_wall = time.perf_counter()  # wall-clock start
                start_cpu = time.thread_time()    # CPU-time start

                color_frame, depth_frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            if color_frame is None or depth_frame is None:
                continue

            # Detection
            detections = detection_xyz(self.model, color_frame, depth_frame, **self.yolo_args)
            if not self.detections_queue.full():
                self.detections_queue.put(detections)

            # Display
            if self.display:
                annotated_frame = draw_detections(color_frame.copy(), detections)
                cv2.imshow("Bounding Box with XYZ", annotated_frame)
                if cv2.waitKey(1) == 27:  # ESC key to stop
                    self.running = False
                    break

            # End timing
            end_wall = time.perf_counter()
            end_cpu = time.thread_time()

            wall_latency_ms = (end_wall - start_wall) * 1000
            cpu_time_ms = (end_cpu - start_cpu) * 1000

            print(f"[Latency] Wall: {wall_latency_ms:.2f} ms | CPU: {cpu_time_ms:.2f} ms")

        cv2.destroyAllWindows()
    
    def start(self, frame_queue: Queue):
        self.frame_queue = frame_queue
        self.running = True
        self.det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.det_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'det_thread'):
            self.det_thread.join()

    def get_latest_detection(self):
        latest_detection = None
        while not self.detections_queue.empty():
            latest_detection = self.detections_queue.get()
        return latest_detection

if __name__ == "__main__":
    pass